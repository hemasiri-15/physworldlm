"""
dataset_gen/generate_entity_dataset.py
───────────────────────────────────────────────────────────────────────────────
Synthetic dataset generator for the PhysWorldLM hierarchical entity ontology
classifier.

PURPOSE
───────
Generates a large JSONL dataset suitable for training a MiniLM-based
hierarchical entity classifier.  The generator does NOT train any model —
it only produces the dataset that a downstream training script will consume.

ONTOLOGY HIERARCHY
───────────────────
physical_entity
│
├── living
│      ├── agent
│      ├── animal
│      └── plant
│
├── rigid_body
│      ├── vehicle
│      ├── robot
│      ├── machine
│      ├── furniture
│      ├── tool
│      ├── container
│      ├── electronic
│      ├── weapon
│      └── sports_object
│
├── environment
│      ├── structure
│      ├── terrain
│      └── fluid
│
├── dynamic_object
│      ├── projectile
│      └── particle
│
└── astronomical
       └── celestial_body

Special class (negative examples):
    non_physical  — abstract concepts with no physical manifestation

OUTPUT RECORD SCHEMA (JSONL)
─────────────────────────────
Each line in the output file is a self-contained JSON object:

{
  "token":                  "Ferrari",
  "entity_type":            "vehicle",
  "parent_class":           "rigid_body",
  "root_class":             "physical_entity",
  "superclass":             "vehicle",
  "subclass":               "car",

  "material":               "steel",
  "phase":                  "solid",
  "mobility":               "self_propelled",
  "size_class":             "large",

  "properties":             ["rigid", "rolling", "motorized"],
  "interaction_properties": ["rigid", "conductive"],
  "affordances":            ["drive", "transport", "collide"],
  "scene_roles":            ["actor", "collider"],
  "aliases":                ["car", "automobile"],

  "confidence":             1.0,
  "negative":               false,
  "possible_classes":       null,
  "variant_of":             "Ferrari",
  "variant_type":           "brand_compound"
}

MULTI-TASK FUTURE COMPATIBILITY
────────────────────────────────
Records are structured to support:
    Entity classification     : token → entity_type
    Material prediction       : token → material
    Phase prediction          : token → phase
    Mobility prediction       : token → mobility
    Size estimation           : token → size_class
    Affordance prediction     : token → affordances
    Property prediction       : token → interaction_properties
    Hierarchical classification: token → parent_class, root_class

USAGE
─────
    python dataset_gen/generate_entity_dataset.py
    python dataset_gen/generate_entity_dataset.py --output datasets/entity_classification.jsonl
    python dataset_gen/generate_entity_dataset.py --samples 50000 --seed 42
    python dataset_gen/generate_entity_dataset.py --balance --quiet

FUTURE PIPELINE
───────────────
The produced dataset feeds directly into:

    MiniLM encoder
         ↓
    Hierarchical classifier  (entity_type  +  parent_class  +  subclass)
         ↓
    Physics ontology resolver
         ↓
    Graph builder  →  WorldSpec
         ↓
    Simulation engine  (Bullet / MuJoCo / Gazebo)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  –  Ontology definition
# ─────────────────────────────────────────────────────────────────────────────

# Every leaf class maps to:
#   (parent_class, root_class)
ONTOLOGY: dict[str, tuple[str, str]] = {
    # living
    "agent":          ("living",        "physical_entity"),
    "animal":         ("living",        "physical_entity"),
    "plant":          ("living",        "physical_entity"),
    # rigid_body
    "vehicle":        ("rigid_body",    "physical_entity"),
    "robot":          ("rigid_body",    "physical_entity"),
    "machine":        ("rigid_body",    "physical_entity"),
    "furniture":      ("rigid_body",    "physical_entity"),
    "tool":           ("rigid_body",    "physical_entity"),
    "container":      ("rigid_body",    "physical_entity"),
    "electronic":     ("rigid_body",    "physical_entity"),
    "weapon":         ("rigid_body",    "physical_entity"),
    "sports_object":  ("rigid_body",    "physical_entity"),
    # environment
    "structure":      ("environment",   "physical_entity"),
    "terrain":        ("environment",   "physical_entity"),
    "fluid":          ("environment",   "physical_entity"),
    # dynamic_object
    "projectile":     ("dynamic_object","physical_entity"),
    "particle":       ("dynamic_object","physical_entity"),
    # astronomical
    "celestial_body": ("astronomical",  "physical_entity"),
    # special negative class — not a physical entity
    "non_physical":   ("abstract",      "non_physical"),
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  –  Physical property taxonomy  (original — preserved verbatim)
#
# Properties drive physics-engine inference:
#   rolling       → wheel friction, rolling resistance
#   motorized     → acceleration constraints, torque limits
#   rigid         → rigid-body collision solver
#   elastic       → restitution coefficient > 0
#   flowing       → fluid-dynamics solver
#   deformable    → soft-body / FEM solver
#   living        → biological-motion model
#   sharp         → penetration / cutting interactions
#   massive       → gravitational dominance
#   conductive    → electrical / thermal simulation
#   container     → interior volume / cargo physics
#   floating      → buoyancy enabled
#   flammable     → combustion model
#   rotating      → angular-momentum tracking
# ─────────────────────────────────────────────────────────────────────────────

PROPERTY_POOL: list[str] = [
    "rigid", "soft", "rolling", "motorized", "living", "flowing",
    "deformable", "flammable", "container", "sharp", "elastic",
    "massive", "rotating", "floating", "conductive",
]

# Canonical property sets per entity_type
# (used for seed tokens; variants may inherit a subset)
ENTITY_PROPERTIES: dict[str, list[str]] = {
    "agent":          ["living", "soft", "rigid"],
    "animal":         ["living", "soft"],
    "plant":          ["living", "soft", "flammable"],

    "vehicle":        ["rigid", "rolling", "motorized"],
    "robot":          ["rigid", "motorized", "conductive"],
    "machine":        ["rigid", "motorized"],
    "furniture":      ["rigid"],
    "tool":           ["rigid", "sharp"],
    "container":      ["rigid", "container"],
    "electronic":     ["rigid", "conductive"],
    "weapon":         ["rigid", "sharp"],
    "sports_object":  ["rigid", "elastic", "rolling"],

    "structure":      ["rigid", "massive"],
    "terrain":        ["rigid", "massive"],
    "fluid":          ["flowing", "deformable"],

    "projectile":     ["rigid", "sharp"],
    "particle":       ["soft", "deformable"],

    "celestial_body": ["rigid", "massive", "rotating"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  –  Hierarchical subclass taxonomy  (original — preserved verbatim)
#
# superclass  = the *coarser* class used by the hierarchical loss
# subclass    = a *finer* subdivision within entity_type
# ─────────────────────────────────────────────────────────────────────────────

# Maps entity_type → list of (subclass, [additional_properties])
SUBCLASS_MAP: dict[str, list[tuple[str, list[str]]]] = {
    "vehicle": [
        ("car",          ["rolling"]),
        ("truck",        ["rolling", "massive"]),
        ("motorcycle",   ["rolling"]),
        ("aircraft",     ["rotating", "floating"]),
        ("watercraft",   ["floating"]),
        ("spacecraft",   ["rotating"]),
        ("rail_vehicle", ["rolling", "massive"]),
        ("construction_vehicle", ["rolling", "massive", "motorized"]),
    ],
    "robot": [
        ("humanoid_robot", ["living"]),
        ("drone",          ["rotating", "floating"]),
        ("rover",          ["rolling"]),
        ("arm_robot",      ["rotating"]),
        ("industrial_robot",["motorized"]),
    ],
    "machine": [
        ("engine",       ["motorized", "flammable"]),
        ("pump",         ["motorized"]),
        ("generator",    ["motorized", "conductive"]),
        ("crane",        ["massive", "motorized"]),
        ("conveyor",     ["rolling", "motorized"]),
    ],
    "furniture": [
        ("seating",      []),
        ("table",        []),
        ("storage",      ["container"]),
        ("bed",          ["soft"]),
        ("shelving",     ["massive"]),
    ],
    "tool": [
        ("cutting_tool", ["sharp"]),
        ("measuring_tool",[]),
        ("fastening_tool",[]),
        ("hand_tool",    []),
        ("power_tool",   ["motorized"]),
    ],
    "container": [
        ("vessel",       ["container"]),
        ("box",          ["container"]),
        ("bag",          ["container", "soft"]),
        ("tank",         ["container", "massive"]),
        ("bottle",       ["container"]),
    ],
    "electronic": [
        ("computing",    ["conductive"]),
        ("communication",["conductive"]),
        ("sensor",       ["conductive"]),
        ("display",      ["conductive"]),
        ("power_device", ["conductive", "flammable"]),
    ],
    "weapon": [
        ("firearm",      ["sharp", "flammable"]),
        ("melee_weapon", ["sharp"]),
        ("explosive",    ["flammable"]),
        ("projectile_weapon", ["sharp"]),
        ("energy_weapon",[]),
    ],
    "sports_object": [
        ("ball",         ["elastic", "rolling"]),
        ("racquet",      ["elastic"]),
        ("stick",        ["rigid"]),
        ("puck",         ["rigid", "rolling"]),
        ("target",       []),
    ],
    "agent": [
        ("human",        ["living"]),
        ("professional", ["living"]),
        ("athlete",      ["living"]),
        ("crowd",        ["living", "massive"]),
    ],
    "animal": [
        ("mammal",       ["living"]),
        ("bird",         ["living", "floating"]),
        ("reptile",      ["living"]),
        ("aquatic",      ["living", "floating"]),
        ("insect",       ["living"]),
    ],
    "plant": [
        ("tree",         ["massive", "flammable"]),
        ("shrub",        ["flammable"]),
        ("grass",        ["soft", "flammable"]),
        ("crop",         ["soft"]),
    ],
    "structure": [
        ("building",     ["massive", "rigid"]),
        ("bridge",       ["massive", "rigid"]),
        ("wall",         ["massive", "rigid"]),
        ("tower",        ["massive", "rigid"]),
        ("barrier",      ["rigid"]),
    ],
    "terrain": [
        ("flat_terrain",  ["massive"]),
        ("elevated_terrain",["massive"]),
        ("water_terrain", ["flowing"]),
        ("synthetic_terrain",["rigid"]),
    ],
    "fluid": [
        ("liquid",        ["flowing", "deformable"]),
        ("gas",           ["flowing", "deformable"]),
        ("plasma",        ["flowing", "deformable", "conductive"]),
        ("molten",        ["flowing", "flammable"]),
    ],
    "projectile": [
        ("ballistic",     ["rigid"]),
        ("guided",        ["rigid", "motorized"]),
        ("thrown",        ["rigid"]),
        ("natural",       ["massive"]),
    ],
    "particle": [
        ("aerosol",       ["flowing"]),
        ("dust",          ["deformable"]),
        ("debris",        ["rigid"]),
        ("quantum",       []),
    ],
    "celestial_body": [
        ("planet",        ["massive", "rotating"]),
        ("moon",          ["massive", "rotating"]),
        ("star",          ["massive", "rotating", "flammable"]),
        ("asteroid",      ["massive", "rigid"]),
        ("comet",         ["massive"]),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  –  Seed vocabulary  (original — preserved verbatim)
# ─────────────────────────────────────────────────────────────────────────────

SEED_VOCAB: dict[str, list[str]] = {

    # ── agent ─────────────────────────────────────────────────────────────────
    "agent": [
        "person", "man", "woman", "human", "people", "individual", "someone",
        "adult", "teenager", "child", "kid", "toddler", "infant", "baby",
        "elder", "senior", "youth", "pedestrian", "bystander", "passenger",
        "driver", "pilot", "captain", "navigator", "operator", "user",
        "worker", "employee", "employer", "manager", "director", "supervisor",
        "engineer", "scientist", "researcher", "professor", "teacher", "student",
        "doctor", "nurse", "surgeon", "paramedic", "firefighter", "police",
        "soldier", "officer", "guard", "security", "detective", "spy",
        "athlete", "runner", "sprinter", "swimmer", "cyclist", "jumper",
        "climber", "skier", "skater", "gymnast", "diver", "boxer",
        "player", "goalkeeper", "striker", "pitcher", "batter",
        "artist", "painter", "sculptor", "musician", "dancer",
        "chef", "baker", "farmer", "fisherman", "miner", "logger",
        "astronaut", "cosmonaut", "commander", "crew", "civilian",
        "humanoid", "android", "cyborg", "avatar", "agent",
        "pedestrian", "commuter", "tourist", "refugee", "survivor",
        "rescuer", "volunteer", "protester", "soldier", "mercenary",
        "acrobat", "performer", "actor", "journalist", "reporter",
        "architect", "designer", "developer", "programmer", "hacker",
        "trader", "investor", "analyst", "economist", "politician",
        "activist", "environmentalist", "explorer", "adventurer",
        "sailor", "surfer", "hiker", "trekker", "mountaineer",
        "skydiver", "parachutist", "racecar driver", "jockey",
        "maintenance worker", "construction worker", "factory worker",
        "delivery person", "courier", "postal worker", "cashier",
        "pharmacist", "dentist", "veterinarian", "therapist",
        "coach", "referee", "umpire", "spectator", "fan",
    ],

    # ── animal ────────────────────────────────────────────────────────────────
    "animal": [
        "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "rabbit",
        "hamster", "guinea pig", "gerbil", "rat", "mouse", "ferret",
        "elephant", "rhinoceros", "hippopotamus", "giraffe", "zebra",
        "lion", "tiger", "leopard", "cheetah", "jaguar", "panther",
        "bear", "wolf", "fox", "coyote", "hyena", "jackal",
        "deer", "moose", "elk", "caribou", "antelope", "gazelle",
        "gorilla", "chimpanzee", "orangutan", "baboon", "monkey",
        "eagle", "hawk", "falcon", "owl", "penguin", "ostrich",
        "parrot", "flamingo", "pelican", "heron", "crane", "stork",
        "sparrow", "robin", "crow", "raven", "pigeon", "dove",
        "snake", "crocodile", "alligator", "lizard", "gecko",
        "turtle", "tortoise", "iguana", "chameleon", "komodo dragon",
        "shark", "whale", "dolphin", "seal", "walrus", "otter",
        "octopus", "squid", "jellyfish", "crab", "lobster", "shrimp",
        "fish", "salmon", "tuna", "goldfish", "koi", "clownfish",
        "frog", "toad", "salamander", "newt", "axolotl",
        "butterfly", "bee", "ant", "beetle", "dragonfly", "moth",
        "spider", "scorpion", "tarantula", "centipede",
        "camel", "llama", "alpaca", "bison", "buffalo", "yak",
        "kangaroo", "koala", "wombat", "platypus", "wallaby",
        "panda", "grizzly bear", "polar bear", "black bear",
        "manta ray", "stingray", "eel", "piranha", "barracuda",
        "cheetah", "chimpanzee", "macaw", "toucan", "peacock",
        "wild boar", "badger", "skunk", "raccoon", "opossum",
    ],

    # ── plant ─────────────────────────────────────────────────────────────────
    "plant": [
        "tree", "oak", "pine", "maple", "birch", "willow", "palm",
        "cedar", "sequoia", "redwood", "bamboo", "eucalyptus",
        "apple tree", "cherry tree", "orange tree", "lemon tree",
        "bush", "shrub", "hedge", "fern", "moss", "lichen",
        "grass", "wheat", "corn", "rice", "sugarcane", "barley",
        "sunflower", "rose", "tulip", "daisy", "orchid", "lily",
        "dandelion", "clover", "ivy", "vine", "cactus", "succulent",
        "seaweed", "kelp", "algae", "plankton",
        "crop", "vegetable", "carrot", "potato", "cabbage", "lettuce",
        "tomato plant", "pepper plant", "bean plant", "pea plant",
        "mangrove", "bonsai", "topiary", "sapling", "seedling",
        "thistle", "nettle", "bracken", "heather", "lavender",
        "mint", "basil", "rosemary", "thyme", "sage",
        "coconut tree", "banana tree", "mango tree", "papaya tree",
        "fig tree", "olive tree", "poplar", "ash tree", "elm tree",
        "thorn bush", "holly", "mistletoe", "pitcher plant",
        "Venus flytrap", "water lily", "lotus", "reed", "cattail",
    ],

    # ── vehicle ───────────────────────────────────────────────────────────────
    "vehicle": [
        "car", "automobile", "sedan", "coupe", "hatchback", "wagon",
        "convertible", "SUV", "crossover", "minivan", "pickup truck",
        "truck", "lorry", "semi-truck", "tanker truck", "dump truck",
        "bus", "minibus", "coach", "school bus", "trolleybus",
        "motorcycle", "motorbike", "scooter", "moped", "dirt bike",
        "bicycle", "e-bike", "tricycle", "quadbike", "ATV",
        "Ferrari", "Tesla", "BMW", "Lamborghini", "Mercedes", "Audi",
        "Porsche", "Bugatti", "Maserati", "Rolls-Royce", "Bentley",
        "Ford", "Chevrolet", "Dodge", "Jeep", "Toyota", "Honda",
        "Volkswagen", "Renault", "Peugeot", "Citroën", "Fiat",
        "tractor", "bulldozer", "excavator", "crane truck", "forklift",
        "ambulance", "fire truck", "police car", "armored vehicle",
        "tank", "military vehicle", "APC", "humvee", "jeep",
        "train", "locomotive", "railcar", "tram", "subway", "metro",
        "monorail", "bullet train", "Shinkansen", "maglev",
        "airplane", "airliner", "fighter jet", "bomber", "biplane",
        "helicopter", "autogyro", "tiltrotor", "Chinook",
        "drone", "UAV", "quadcopter", "octocopter", "fixed-wing drone",
        "spaceship", "rocket", "space shuttle", "capsule", "spacecraft",
        "satellite", "space station", "lander", "rover",
        "boat", "motorboat", "sailboat", "canoe", "kayak", "raft",
        "yacht", "catamaran", "trimaran", "hovercraft",
        "ship", "cargo ship", "container ship", "tanker", "cruise ship",
        "submarine", "naval vessel", "destroyer", "aircraft carrier",
        "Cybertruck", "Rivian", "Hummer", "Land Rover", "Range Rover",
        "snowmobile", "jet ski", "segway", "electric scooter",
    ],

    # ── robot ─────────────────────────────────────────────────────────────────
    "robot": [
        "robot", "android", "humanoid robot", "bipedal robot",
        "Boston Dynamics robot", "Atlas robot", "Spot robot",
        "Optimus robot", "ASIMO", "Pepper robot", "NAO robot",
        "drone", "combat drone", "delivery drone", "racing drone",
        "quadcopter", "hexacopter", "surveillance drone",
        "rover", "Mars rover", "planetary rover", "lunar rover",
        "Curiosity rover", "Perseverance rover",
        "robotic arm", "industrial arm", "KUKA arm", "ABB robot",
        "welding robot", "painting robot", "assembly robot",
        "surgical robot", "Da Vinci robot", "medical robot",
        "autonomous vehicle", "self-driving car", "robot car",
        "warehouse robot", "Kiva robot", "Amazon robot",
        "exoskeleton", "powered suit", "robotic suit",
        "swarm robot", "nano-robot", "microbot",
        "chatbot", "AI assistant", "digital agent",
        "robotic dog", "robot pet", "mechanical animal",
        "underwater robot", "AUV", "ROV", "submersible robot",
        "bomb disposal robot", "EOD robot", "tactical robot",
        "space robot", "astronaut robot", "robonaut",
        "cleaning robot", "Roomba", "floor robot",
        "agricultural robot", "harvesting robot", "planting robot",
        "construction robot", "bricklaying robot", "3D printing robot",
    ],

    # ── machine ───────────────────────────────────────────────────────────────
    "machine": [
        "engine", "motor", "generator", "turbine", "compressor",
        "pump", "hydraulic pump", "pneumatic pump",
        "lathe", "milling machine", "CNC machine", "drill press",
        "press", "hydraulic press", "stamping press", "punch press",
        "conveyor belt", "conveyor system", "assembly line",
        "crane", "overhead crane", "tower crane", "gantry crane",
        "elevator", "escalator", "lift", "moving walkway",
        "treadmill", "elliptical", "stationary bike", "rowing machine",
        "washing machine", "dryer", "dishwasher", "vacuum cleaner",
        "air conditioner", "HVAC system", "heat pump", "boiler",
        "printing press", "3D printer", "laser cutter", "plotter",
        "ATM", "vending machine", "ticket machine", "kiosk",
        "slot machine", "pinball machine", "arcade machine",
        "wind turbine", "water turbine", "steam turbine",
        "nuclear reactor", "power plant", "transformer",
        "satellite dish", "radar array", "radio telescope",
        "centrifuge", "autoclave", "spectrometer", "electron microscope",
        "MRI scanner", "CT scanner", "X-ray machine", "ultrasound",
        "forge", "blast furnace", "kiln", "smelter", "foundry",
        "oil rig", "drilling rig", "mining machine", "tunneling machine",
        "agricultural machine", "combine harvester", "thresher",
        "loom", "spinning wheel", "textile machine", "knitting machine",
    ],

    # ── furniture ─────────────────────────────────────────────────────────────
    "furniture": [
        "chair", "armchair", "recliner", "rocking chair", "office chair",
        "stool", "barstool", "bench", "ottoman", "footstool",
        "sofa", "couch", "loveseat", "sectional", "futon",
        "table", "dining table", "coffee table", "side table", "console table",
        "desk", "writing desk", "standing desk", "drafting table",
        "bed", "single bed", "double bed", "king bed", "bunk bed",
        "cot", "daybed", "crib", "cradle", "hammock",
        "bookshelf", "bookcase", "shelving unit", "display shelf",
        "cabinet", "filing cabinet", "storage cabinet", "medicine cabinet",
        "wardrobe", "closet", "dresser", "chest of drawers", "nightstand",
        "sideboard", "buffet", "hutch", "credenza",
        "mirror", "vanity", "dressing table", "makeup table",
        "lamp", "floor lamp", "table lamp", "desk lamp",
        "rug", "carpet", "mat", "doormat",
        "curtain", "blind", "shutters", "screen divider",
        "coat rack", "hat stand", "umbrella stand",
        "TV stand", "entertainment center", "media console",
        "bar cart", "serving trolley", "plant stand",
        "folding table", "folding chair", "picnic table",
    ],

    # ── tool ──────────────────────────────────────────────────────────────────
    "tool": [
        "hammer", "mallet", "sledgehammer",
        "screwdriver", "Phillips screwdriver", "flathead screwdriver",
        "wrench", "socket wrench", "Allen key", "torque wrench",
        "pliers", "needle-nose pliers", "wire cutters", "clamp",
        "saw", "handsaw", "hacksaw", "jigsaw", "circular saw",
        "drill", "power drill", "cordless drill", "impact driver",
        "chisel", "gouge", "punch", "awl",
        "knife", "utility knife", "box cutter", "penknife",
        "scissors", "shears", "pruning shears", "snips",
        "tape measure", "ruler", "level", "square", "protractor",
        "caliper", "micrometer", "gauge", "vernier",
        "soldering iron", "heat gun", "blowtorch",
        "paintbrush", "roller", "sprayer", "spatula",
        "trowel", "float", "putty knife", "scraper",
        "shovel", "spade", "pitchfork", "rake", "hoe",
        "axe", "hatchet", "machete", "crowbar", "pry bar",
        "ladder", "step ladder", "extension ladder",
        "staple gun", "nail gun", "brad nailer",
        "multimeter", "oscilloscope", "spectrum analyzer",
        "magnifying glass", "loupe", "microscope",
        "plunger", "pipe wrench", "tubing cutter",
        "grinder", "angle grinder", "bench grinder",
    ],

    # ── container ─────────────────────────────────────────────────────────────
    "container": [
        "box", "cardboard box", "wooden box", "metal box",
        "crate", "wooden crate", "plastic crate", "pallet",
        "bag", "plastic bag", "paper bag", "grocery bag", "trash bag",
        "backpack", "rucksack", "duffle bag", "briefcase", "suitcase",
        "bottle", "glass bottle", "plastic bottle", "water bottle",
        "can", "tin can", "aluminum can", "spray can",
        "jar", "mason jar", "glass jar", "pickle jar",
        "barrel", "oil barrel", "wine barrel", "drum",
        "tank", "fuel tank", "water tank", "storage tank",
        "bucket", "pail", "tub", "basin",
        "pot", "pan", "saucepan", "wok", "pressure cooker",
        "cup", "mug", "glass", "goblet", "tumbler",
        "bowl", "salad bowl", "mixing bowl", "soup bowl",
        "envelope", "package", "parcel", "tube",
        "bin", "trash bin", "recycling bin", "waste bin",
        "capsule", "pod", "shell", "casing",
        "vault", "safe", "chest", "strongbox", "lockbox",
        "pitcher", "jug", "ewer", "vase", "urn",
        "lunchbox", "toolbox", "tackle box", "storage box",
        "shipping container", "intermodal container",
        "fuel canister", "propane tank", "oxygen cylinder",
    ],

    # ── electronic ────────────────────────────────────────────────────────────
    "electronic": [
        "smartphone", "iPhone", "Android phone", "mobile phone",
        "tablet", "iPad", "Android tablet", "e-reader", "Kindle",
        "laptop", "MacBook", "notebook", "netbook", "Chromebook",
        "desktop computer", "PC", "Mac", "workstation", "server",
        "television", "TV", "smart TV", "monitor", "display", "screen",
        "camera", "DSLR", "mirrorless camera", "action camera", "GoPro",
        "video camera", "webcam", "dashcam", "security camera",
        "headphones", "earbuds", "AirPods", "earphones", "speaker",
        "soundbar", "subwoofer", "amplifier", "receiver",
        "router", "modem", "switch", "hub", "access point",
        "smartwatch", "fitness tracker", "smart band", "GPS watch",
        "GPS device", "navigation system", "dashboard GPS",
        "drone controller", "game controller", "joystick", "keyboard",
        "mouse", "trackpad", "graphics tablet", "stylus",
        "printer", "laser printer", "inkjet printer", "scanner",
        "projector", "beamer", "document camera",
        "microphone", "condenser mic", "dynamic mic", "lapel mic",
        "radio", "walkie-talkie", "transceiver", "CB radio",
        "calculator", "scientific calculator", "graphing calculator",
        "sensor", "lidar", "sonar", "radar", "thermometer",
        "smoke detector", "motion sensor", "proximity sensor",
        "battery pack", "power bank", "charger", "inverter",
        "solar panel", "LED strip", "smart bulb", "smart plug",
        "VR headset", "AR glasses", "mixed reality device",
    ],

    # ── weapon ────────────────────────────────────────────────────────────────
    "weapon": [
        "gun", "pistol", "revolver", "handgun", "sidearm",
        "rifle", "assault rifle", "AK-47", "M16", "AR-15",
        "shotgun", "pump-action shotgun", "double-barrel shotgun",
        "sniper rifle", "marksman rifle", "bolt-action rifle",
        "machine gun", "LMG", "HMG", "minigun",
        "submachine gun", "MP5", "Uzi",
        "crossbow", "compound bow", "longbow", "recurve bow",
        "sword", "longsword", "katana", "rapier", "scimitar",
        "dagger", "knife", "combat knife", "bayonet", "stiletto",
        "axe", "battle axe", "tomahawk", "hatchet",
        "spear", "javelin", "lance", "pike", "halberd",
        "mace", "flail", "war hammer", "club", "baton",
        "grenade", "frag grenade", "smoke grenade", "flashbang",
        "rocket launcher", "RPG", "bazooka", "anti-tank weapon",
        "missile launcher", "surface-to-air missile", "SAM",
        "mine", "landmine", "naval mine", "IED",
        "cannon", "howitzer", "artillery", "mortar",
        "tank gun", "naval gun", "railgun",
        "taser", "stun gun", "pepper spray", "tear gas",
        "flamethrower", "incendiary device", "Molotov cocktail",
        "bomb", "aerial bomb", "bunker buster", "cruise missile",
        "nuclear warhead", "ICBM", "ballistic missile",
        "nunchaku", "sai", "shuriken", "throwing star",
    ],

    # ── sports_object ─────────────────────────────────────────────────────────
    "sports_object": [
        "football", "soccer ball", "basketball", "volleyball",
        "baseball", "softball", "cricket ball", "tennis ball",
        "golf ball", "ping pong ball", "squash ball", "racquetball",
        "rugby ball", "American football", "frisbee", "disc",
        "tennis racquet", "badminton racquet", "squash racquet",
        "baseball bat", "cricket bat", "hockey stick", "lacrosse stick",
        "golf club", "driver", "iron", "putter", "wedge",
        "puck", "hockey puck", "roller derby",
        "skateboard", "longboard", "penny board", "surfboard",
        "snowboard", "ski", "water ski", "wakeboard",
        "kayak paddle", "canoe paddle", "oar", "rowing blade",
        "javelin", "discus", "shot put", "hammer throw",
        "pole vault pole", "high jump bar", "hurdle",
        "boxing gloves", "punching bag", "speed bag",
        "dumbbell", "barbell", "kettlebell", "weight plate",
        "resistance band", "jump rope", "pull-up bar",
        "yoga mat", "exercise ball", "foam roller",
        "archery bow", "quiver", "target",
        "goal post", "net", "basket", "hoop",
        "dartboard", "dart", "billiard ball", "cue stick",
        "bowling ball", "bowling pin", "lane",
    ],

    # ── structure ─────────────────────────────────────────────────────────────
    "structure": [
        "wall", "brick wall", "concrete wall", "retaining wall",
        "building", "skyscraper", "tower", "high-rise", "apartment",
        "house", "bungalow", "mansion", "villa", "cottage", "cabin",
        "office building", "commercial building", "warehouse", "factory",
        "bridge", "suspension bridge", "arch bridge", "cable bridge",
        "overpass", "flyover", "viaduct", "aqueduct",
        "dam", "reservoir dam", "hydroelectric dam",
        "tunnel", "underpass", "subway tunnel", "road tunnel",
        "fence", "chain-link fence", "wooden fence", "barbed wire",
        "gate", "iron gate", "security gate", "barrier",
        "pillar", "column", "beam", "girder", "rafter",
        "staircase", "escalator housing", "elevator shaft",
        "platform", "loading dock", "jetty", "pier", "wharf",
        "lighthouse", "watchtower", "control tower",
        "chimney", "smokestack", "cooling tower",
        "antenna tower", "radio mast", "cell tower", "pylon",
        "silo", "grain silo", "storage silo",
        "greenhouse", "hangar", "barn", "stable",
        "monument", "statue", "obelisk", "arch", "triumphal arch",
        "stadium", "arena", "amphitheater", "grandstand",
        "runway", "taxiway", "control tower", "terminal building",
        "power station", "substation", "transformer station",
        "water tower", "oil derrick", "drilling platform",
    ],

    # ── terrain ───────────────────────────────────────────────────────────────
    "terrain": [
        "ground", "earth", "soil", "dirt", "mud",
        "road", "highway", "motorway", "expressway", "freeway",
        "street", "avenue", "boulevard", "lane", "alley",
        "path", "trail", "footpath", "sidewalk", "pavement",
        "track", "dirt track", "race track", "running track",
        "hill", "hillside", "slope", "gradient", "incline",
        "mountain", "peak", "summit", "ridge", "cliff",
        "valley", "canyon", "gorge", "ravine", "gully",
        "plain", "flatland", "prairie", "steppe", "savannah",
        "desert", "sand dune", "salt flat", "rocky terrain",
        "forest floor", "jungle floor", "undergrowth",
        "field", "farmland", "pasture", "meadow", "grassland",
        "beach", "shoreline", "coastline", "tidal flat",
        "wetland", "marsh", "swamp", "bog", "fen",
        "ice", "glacier", "ice field", "frozen lake", "tundra",
        "snow", "snowfield", "avalanche debris",
        "gravel", "rubble", "debris field", "ruins",
        "runway", "airstrip", "taxiway", "apron",
        "parking lot", "parking garage floor", "courtyard",
        "rooftop", "roof surface", "platform surface",
        "seabed", "ocean floor", "riverbed", "lakebed",
        "lava field", "volcanic terrain", "crater",
    ],

    # ── fluid ─────────────────────────────────────────────────────────────────
    "fluid": [
        "water", "freshwater", "saltwater", "seawater", "distilled water",
        "river", "stream", "creek", "brook", "tributary",
        "lake", "pond", "reservoir", "ocean", "sea", "bay",
        "oil", "crude oil", "motor oil", "hydraulic oil", "cooking oil",
        "gasoline", "petrol", "diesel", "kerosene", "jet fuel",
        "blood", "plasma", "lymph", "cerebrospinal fluid",
        "milk", "cream", "buttermilk", "yogurt liquid",
        "wine", "beer", "juice", "tea", "coffee",
        "acid", "hydrochloric acid", "sulfuric acid", "nitric acid",
        "base", "sodium hydroxide", "ammonia solution",
        "alcohol", "ethanol", "methanol", "isopropyl alcohol",
        "ink", "paint", "varnish", "lacquer", "resin",
        "glue", "adhesive", "epoxy", "resin",
        "lava", "magma", "molten rock", "molten metal",
        "molten steel", "molten iron", "molten glass", "molten lead",
        "honey", "syrup", "molasses", "treacle", "glycerin",
        "mercury", "liquid nitrogen", "liquid oxygen", "liquid helium",
        "steam", "vapor", "mist", "fog", "cloud",
        "rain", "hail", "sleet", "drizzle",
        "mud", "slurry", "sludge", "quicksand",
        "gas", "air", "oxygen", "nitrogen", "helium", "hydrogen",
        "smoke", "exhaust", "fumes", "aerosol", "spray",
        "coolant", "antifreeze", "brake fluid", "transmission fluid",
    ],

    # ── projectile ────────────────────────────────────────────────────────────
    "projectile": [
        "bullet", "round", "cartridge", "slug", "buckshot",
        "arrow", "bolt", "crossbow bolt", "broadhead",
        "missile", "cruise missile", "ballistic missile", "ICBM",
        "rocket", "unguided rocket", "spin-stabilized rocket",
        "dart", "blowgun dart", "tranquilizer dart",
        "cannonball", "grapeshot", "chainshot",
        "grenade", "mortar round", "artillery shell",
        "bomb", "aerial bomb", "smart bomb", "gravity bomb",
        "torpedo", "naval torpedo", "self-propelled torpedo",
        "stone", "rock", "thrown rock", "sling stone",
        "spear", "javelin", "throwing spear", "harpoon",
        "shuriken", "throwing star", "kunai",
        "disc", "flying disc", "frisbee throw",
        "pellet", "BB pellet", "airsoft pellet", "paintball",
        "meteor", "meteoroid", "meteorite", "bolide",
        "asteroid fragment", "cosmic debris",
        "snowball", "ice ball", "frozen projectile",
        "flare", "signal flare", "parachute flare",
        "tracer round", "armor-piercing round", "explosive round",
        "ball bearing", "steel ball", "slingshot projectile",
        "plasma bolt", "laser pulse", "railgun projectile",
        "net", "bola", "grappling hook",
    ],

    # ── particle ──────────────────────────────────────────────────────────────
    "particle": [
        "dust", "dust particle", "fine dust", "PM2.5", "PM10",
        "smoke particle", "soot", "ash", "char",
        "sand particle", "grain of sand", "silica particle",
        "pollen", "spore", "seed", "airborne seed",
        "aerosol", "droplet", "mist droplet", "spray droplet",
        "debris", "micro-debris", "space debris", "orbital debris",
        "snowflake", "ice crystal", "frost crystal", "hailstone",
        "spark", "ember", "cinder", "burning particle",
        "chip", "flake", "fragment", "shard", "sliver",
        "bubble", "foam bubble", "cavitation bubble",
        "droplet", "raindrop", "condensation droplet",
        "proton", "neutron", "electron", "photon",
        "quark", "lepton", "boson", "fermion",
        "ion", "plasma particle", "charged particle",
        "nanoparticle", "microparticle", "quantum dot",
        "molecule", "atom", "radical", "cluster",
        "sediment", "silt", "clay particle", "colloid",
        "fiber", "micro-fiber", "carbon fiber fragment",
        "paint chip", "rust flake", "corrosion particle",
    ],

    # ── celestial_body ────────────────────────────────────────────────────────
    "celestial_body": [
        "planet", "terrestrial planet", "gas giant", "ice giant",
        "Earth", "Mars", "Venus", "Mercury", "Jupiter", "Saturn",
        "Uranus", "Neptune", "Pluto", "dwarf planet",
        "moon", "natural satellite", "Europa", "Titan", "Ganymede",
        "Io", "Callisto", "Enceladus", "Triton", "Charon",
        "star", "sun", "yellow dwarf", "red giant", "white dwarf",
        "neutron star", "pulsar", "magnetar", "binary star",
        "red dwarf", "blue giant", "supergiant", "T Tauri star",
        "asteroid", "near-Earth asteroid", "main-belt asteroid",
        "Ceres", "Vesta", "Pallas", "Hygiea",
        "comet", "short-period comet", "long-period comet",
        "Halley's Comet", "Hale-Bopp", "Oumuamua",
        "meteor", "meteoroid", "fireball", "bolide",
        "black hole", "stellar black hole", "supermassive black hole",
        "galaxy", "Milky Way", "Andromeda", "spiral galaxy",
        "nebula", "planetary nebula", "emission nebula",
        "galaxy cluster", "quasar", "pulsar", "magnetar",
        "space station", "ISS", "Mir", "Tiangong",
        "space probe", "Voyager", "Cassini", "New Horizons",
        "artificial satellite", "GPS satellite", "spy satellite",
        "dark matter halo", "interstellar object",
    ],

    # ── non_physical  (negative examples — abstract concepts) ─────────────────
    "non_physical": [
        # emotions & mental states
        "love", "hate", "anger", "fear", "joy", "sadness", "jealousy",
        "happiness", "grief", "anxiety", "depression", "excitement",
        "boredom", "loneliness", "pride", "shame", "guilt", "hope",
        # abstract social concepts
        "democracy", "freedom", "justice", "equality", "liberty",
        "authority", "power", "politics", "government", "law",
        "culture", "tradition", "ideology", "philosophy", "religion",
        "morality", "ethics", "virtue", "sin", "belief",
        # cognitive / informational
        "thought", "idea", "concept", "knowledge", "memory",
        "imagination", "consciousness", "awareness", "intelligence",
        "logic", "reasoning", "creativity", "intuition", "wisdom",
        "algorithm", "software", "program", "code", "function",
        "data", "information", "signal", "pattern", "model",
        "theory", "hypothesis", "proof", "theorem", "axiom",
        # aesthetic / cultural
        "beauty", "art", "music", "poetry", "narrative", "story",
        "myth", "legend", "humor", "irony", "metaphor",
        # economic / social processes
        "economics", "capitalism", "socialism", "currency", "value",
        "trade", "market", "inflation", "recession", "profit",
        "democracy", "election", "vote", "policy", "law",
        # mathematics / formal systems
        "mathematics", "geometry", "calculus", "algebra", "statistics",
        "probability", "infinity", "symmetry", "topology",
        # language
        "language", "grammar", "syntax", "semantics", "word",
        "sentence", "meaning", "definition", "metaphor",
        # time / space as abstract
        "time", "duration", "moment", "eternity", "past", "future",
        "space", "dimension", "distance", "direction",
        # miscellaneous abstract
        "relationship", "communication", "trust", "friendship",
        "community", "society", "civilization", "history",
        "change", "growth", "decay", "entropy", "chaos", "order",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  –  Adjective and modifier banks for variant generation
#               (original — preserved verbatim)
# ─────────────────────────────────────────────────────────────────────────────

ADJECTIVES: dict[str, list[str]] = {
    "colour": [
        "red", "blue", "green", "black", "white", "yellow", "orange",
        "grey", "silver", "gold", "purple", "pink", "brown", "cyan",
        "dark", "bright", "light", "neon",
    ],
    "size": [
        "large", "small", "huge", "tiny", "massive", "miniature",
        "giant", "micro", "nano", "compact", "full-size",
    ],
    "material": [
        "wooden", "steel", "metal", "plastic", "rubber", "glass",
        "concrete", "carbon-fiber", "titanium", "ceramic", "stone",
        "aluminum", "iron", "copper", "bronze",
    ],
    "state": [
        "heavy", "light", "fast", "slow", "old", "new", "modern",
        "ancient", "broken", "damaged", "intact", "hollow",
    ],
    "domain": {
        "vehicle":       ["electric", "hybrid", "autonomous", "armored", "flying"],
        "robot":         ["autonomous", "bipedal", "wheeled", "military"],
        "weapon":        ["automatic", "semi-automatic", "manual", "guided"],
        "sports_object": ["inflatable", "regulation", "official"],
        "electronic":    ["wireless", "portable", "wearable", "smart"],
        "tool":          ["power", "hand", "precision", "heavy-duty"],
        "fluid":         ["hot", "cold", "pressurized", "viscous", "corrosive"],
        "structure":     ["reinforced", "prefabricated", "underground", "aerial"],
        "terrain":       ["rough", "smooth", "wet", "frozen", "sandy", "rocky"],
        "agent":         ["trained", "armed", "uniformed", "civilian"],
        "animal":        ["wild", "domestic", "trained", "endangered"],
        "projectile":    ["guided", "armor-piercing", "explosive", "incendiary"],
    },
}

# Compound noun modifiers per entity_type
COMPOUND_MODIFIERS: dict[str, list[str]] = {
    "vehicle":       ["sports", "race", "street", "off-road", "all-terrain",
                      "electric", "hydrogen", "hybrid", "self-driving", "luxury"],
    "robot":         ["battle", "rescue", "service", "companion", "patrol"],
    "tool":          ["power", "hand", "cutting", "measuring", "fastening"],
    "container":     ["storage", "shipping", "cargo", "sealed", "pressurized"],
    "electronic":    ["portable", "wearable", "smart", "wireless", "solar"],
    "weapon":        ["assault", "sniper", "anti-tank", "anti-aircraft"],
    "structure":     ["suspension", "steel-frame", "reinforced-concrete"],
    "furniture":     ["office", "outdoor", "folding", "adjustable", "ergonomic"],
    "sports_object": ["professional", "training", "competition", "regulation"],
    "fluid":         ["pressurized", "filtered", "recycled"],
    "terrain":       ["urban", "off-road", "mountainous", "coastal", "arctic"],
    "agent":         ["emergency", "first-response", "combat", "undercover"],
    "projectile":    ["guided", "ballistic", "armor-piercing"],
    "animal":        ["pack", "herd", "flock", "school"],
    "plant":         ["tropical", "alpine", "desert", "aquatic"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  –  Ambiguous examples  (original — preserved verbatim)
# ─────────────────────────────────────────────────────────────────────────────

AMBIGUOUS_EXAMPLES: list[dict] = [
    {"token": "drone",         "entity_type": "robot",   "confidence": 0.75,
     "possible_classes": ["robot", "vehicle"]},
    {"token": "satellite",     "entity_type": "machine", "confidence": 0.70,
     "possible_classes": ["machine", "celestial_body", "electronic"]},
    {"token": "spaceship",     "entity_type": "vehicle", "confidence": 0.80,
     "possible_classes": ["vehicle", "structure"]},
    {"token": "robot dog",     "entity_type": "robot",   "confidence": 0.85,
     "possible_classes": ["robot", "animal"]},
    {"token": "AI assistant",  "entity_type": "robot",   "confidence": 0.60,
     "possible_classes": ["robot", "non_physical"]},
    {"token": "smartwatch",    "entity_type": "electronic", "confidence": 0.80,
     "possible_classes": ["electronic", "tool"]},
    {"token": "meteor",        "entity_type": "projectile", "confidence": 0.70,
     "possible_classes": ["projectile", "celestial_body"]},
    {"token": "river",         "entity_type": "fluid",   "confidence": 0.75,
     "possible_classes": ["fluid", "terrain"]},
    {"token": "moon",          "entity_type": "celestial_body", "confidence": 0.90,
     "possible_classes": ["celestial_body"]},
    {"token": "dust",          "entity_type": "particle","confidence": 0.72,
     "possible_classes": ["particle", "terrain"]},
    {"token": "submarine",     "entity_type": "vehicle", "confidence": 0.85,
     "possible_classes": ["vehicle", "weapon"]},
    {"token": "exoskeleton",   "entity_type": "robot",   "confidence": 0.72,
     "possible_classes": ["robot", "tool"]},
    {"token": "tree",          "entity_type": "plant",   "confidence": 0.88,
     "possible_classes": ["plant", "structure"]},
    {"token": "rock",          "entity_type": "terrain", "confidence": 0.65,
     "possible_classes": ["terrain", "projectile"]},
    {"token": "snowball",      "entity_type": "projectile", "confidence": 0.78,
     "possible_classes": ["projectile", "particle"]},
    {"token": "water balloon", "entity_type": "container","confidence": 0.68,
     "possible_classes": ["container", "projectile"]},
    {"token": "javelin",       "entity_type": "projectile","confidence": 0.80,
     "possible_classes": ["projectile", "sports_object", "weapon"]},
    {"token": "grenade",       "entity_type": "weapon",  "confidence": 0.82,
     "possible_classes": ["weapon", "projectile"]},
    {"token": "fog",           "entity_type": "fluid",   "confidence": 0.70,
     "possible_classes": ["fluid", "particle"]},
    {"token": "cloud",         "entity_type": "fluid",   "confidence": 0.65,
     "possible_classes": ["fluid", "particle"]},
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  –  Hard examples  (original — preserved verbatim)
# ─────────────────────────────────────────────────────────────────────────────

HARD_EXAMPLES: list[tuple[str, str]] = [
    ("bookshelf",           "structure"),
    ("smartphone",          "electronic"),
    ("submarine",           "vehicle"),
    ("spaceship",           "vehicle"),
    ("river",               "fluid"),
    ("road",                "terrain"),
    ("robot",               "robot"),
    ("drone",               "robot"),
    ("meteor",              "projectile"),
    ("pebble",              "projectile"),
    ("AK-47",               "weapon"),
    ("BMW",                 "vehicle"),
    ("Roomba",              "robot"),
    ("Segway",              "vehicle"),
    ("Hubble",              "machine"),
    ("ISS",                 "structure"),
    ("fire",                "fluid"),
    ("lava",                "fluid"),
    ("glacier",             "terrain"),
    ("tornado",             "fluid"),
    ("solar panel",         "electronic"),
    ("crane",               "machine"),
    ("excavator",           "vehicle"),
    ("ambulance",           "vehicle"),
    ("staircase",           "structure"),
    ("parking lot",         "terrain"),
    ("backpack",            "container"),
    ("water tower",         "structure"),
    ("combine harvester",   "machine"),
    ("flamethrower",        "weapon"),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  –  NEW: Material ontology
#
# Maps entity_type (and optionally subclass keyword) to likely primary
# material.  The resolver checks subclass keywords first, then falls back
# to the entity_type default.
#
# Supported materials:
#   steel  iron  aluminum  wood  plastic  rubber  glass  paper  fabric
#   concrete  stone  water  ice  sand  soil  flesh  ceramic  carbon_fiber
#   mixed  unknown
# ─────────────────────────────────────────────────────────────────────────────

# material values used across the codebase
MATERIALS = [
    "steel", "iron", "aluminum", "wood", "plastic", "rubber", "glass",
    "paper", "fabric", "concrete", "stone", "water", "ice", "sand",
    "soil", "flesh", "ceramic", "carbon_fiber", "mixed", "unknown",
]

# Default material per entity_type (used when subclass keyword matching fails)
ENTITY_MATERIALS: dict[str, str] = {
    "agent":          "flesh",
    "animal":         "flesh",
    "plant":          "wood",
    "vehicle":        "steel",
    "robot":          "steel",
    "machine":        "steel",
    "furniture":      "wood",
    "tool":           "steel",
    "container":      "plastic",
    "electronic":     "plastic",
    "weapon":         "steel",
    "sports_object":  "rubber",
    "structure":      "concrete",
    "terrain":        "soil",
    "fluid":          "water",
    "projectile":     "steel",
    "particle":       "mixed",
    "celestial_body": "stone",
    "non_physical":   "unknown",
}

# Keyword → material overrides; checked against token.lower()
MATERIAL_KEYWORD_OVERRIDES: dict[str, str] = {
    # materials explicitly named
    "wooden":     "wood",   "oak":      "wood",   "pine":     "wood",
    "maple":      "wood",   "bamboo":   "wood",   "timber":   "wood",
    "steel":      "steel",  "iron":     "iron",   "aluminum": "aluminum",
    "titanium":   "steel",  "alloy":    "steel",
    "plastic":    "plastic","rubber":   "rubber", "vinyl":    "plastic",
    "glass":      "glass",  "crystal":  "glass",
    "concrete":   "concrete","cement":  "concrete","brick":   "concrete",
    "stone":      "stone",  "rock":     "stone",  "granite":  "stone",
    "marble":     "stone",  "slate":    "stone",
    "fabric":     "fabric", "cloth":    "fabric", "textile":  "fabric",
    "leather":    "fabric", "silk":     "fabric", "wool":     "fabric",
    "paper":      "paper",  "cardboard":"paper",
    "ceramic":    "ceramic","porcelain":"ceramic","clay":     "ceramic",
    "carbon":     "carbon_fiber",
    # entity-specific keywords
    "water":      "water",  "river":    "water",  "ocean":    "water",
    "lake":       "water",  "rain":     "water",  "blood":    "water",
    "ice":        "ice",    "glacier":  "ice",    "snow":     "ice",
    "sand":       "sand",   "dust":     "sand",   "soil":     "soil",
    "dirt":       "soil",   "mud":      "soil",   "earth":    "soil",
    "human":      "flesh",  "person":   "flesh",  "man":      "flesh",
    "woman":      "flesh",  "child":    "flesh",  "animal":   "flesh",
    "dog":        "flesh",  "cat":      "flesh",  "horse":    "flesh",
    "ball":       "rubber", "tyre":     "rubber", "tire":     "rubber",
    "bottle":     "glass",  "jar":      "glass",  "cup":      "glass",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  –  NEW: Phase ontology
#
# Solid / liquid / gas / plasma / granular
# ─────────────────────────────────────────────────────────────────────────────

PHASES = ["solid", "liquid", "gas", "plasma", "granular"]

# Default phase per entity_type
ENTITY_PHASES: dict[str, str] = {
    "agent":          "solid",
    "animal":         "solid",
    "plant":          "solid",
    "vehicle":        "solid",
    "robot":          "solid",
    "machine":        "solid",
    "furniture":      "solid",
    "tool":           "solid",
    "container":      "solid",
    "electronic":     "solid",
    "weapon":         "solid",
    "sports_object":  "solid",
    "structure":      "solid",
    "terrain":        "solid",
    "fluid":          "liquid",
    "projectile":     "solid",
    "particle":       "granular",
    "celestial_body": "solid",
    "non_physical":   "unknown",
}

# Keyword → phase overrides; checked against token.lower()
PHASE_KEYWORD_OVERRIDES: dict[str, str] = {
    "water":   "liquid", "river":   "liquid", "lake":    "liquid",
    "ocean":   "liquid", "oil":     "liquid", "blood":   "liquid",
    "lava":    "liquid", "magma":   "liquid", "mercury": "liquid",
    "honey":   "liquid", "milk":    "liquid", "acid":    "liquid",
    "paint":   "liquid", "glue":    "liquid", "syrup":   "liquid",
    "steam":   "gas",    "vapor":   "gas",    "fog":     "gas",
    "smoke":   "gas",    "air":     "gas",    "gas":     "gas",
    "cloud":   "gas",    "mist":    "gas",    "fumes":   "gas",
    "helium":  "gas",    "oxygen":  "gas",    "hydrogen":"gas",
    "nitrogen":"gas",    "exhaust": "gas",
    "fire":    "plasma", "plasma":  "plasma", "lightning":"plasma",
    "arc":     "plasma",
    "dust":    "granular","sand":   "granular","ash":    "granular",
    "powder":  "granular","grain":  "granular","soot":   "granular",
    "snow":    "granular","pollen": "granular",
    "ice":     "solid",  "glacier": "solid",  "crystal": "solid",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10  –  NEW: Mobility ontology
#
# static / movable / self_propelled / flowing / flying
# ─────────────────────────────────────────────────────────────────────────────

MOBILITY_VALUES = ["static", "movable", "self_propelled", "flowing", "flying"]

# Default mobility per entity_type
ENTITY_MOBILITY: dict[str, str] = {
    "agent":          "self_propelled",
    "animal":         "self_propelled",
    "plant":          "static",
    "vehicle":        "self_propelled",
    "robot":          "self_propelled",
    "machine":        "static",
    "furniture":      "movable",
    "tool":           "movable",
    "container":      "movable",
    "electronic":     "movable",
    "weapon":         "movable",
    "sports_object":  "movable",
    "structure":      "static",
    "terrain":        "static",
    "fluid":          "flowing",
    "projectile":     "self_propelled",
    "particle":       "flowing",
    "celestial_body": "self_propelled",
    "non_physical":   "unknown",
}

# Keyword → mobility overrides
MOBILITY_KEYWORD_OVERRIDES: dict[str, str] = {
    "wall":       "static",   "building":   "static",   "bridge":   "static",
    "road":       "static",   "mountain":   "static",   "terrain":  "static",
    "ground":     "static",   "floor":      "static",   "structure":"static",
    "river":      "flowing",  "water":      "flowing",  "lava":     "flowing",
    "smoke":      "flowing",  "gas":        "flowing",  "fluid":    "flowing",
    "steam":      "flowing",  "air":        "flowing",  "cloud":    "flowing",
    "bird":       "flying",   "eagle":      "flying",   "hawk":     "flying",
    "drone":      "flying",   "aircraft":   "flying",   "jet":      "flying",
    "helicopter": "flying",   "airplane":   "flying",   "rocket":   "flying",
    "missile":    "flying",   "butterfly":  "flying",
    "table":      "movable",  "chair":      "movable",  "box":      "movable",
    "ball":       "movable",  "bottle":     "movable",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11  –  NEW: Size ontology
#
# tiny / small / medium / large / huge / astronomical
# ─────────────────────────────────────────────────────────────────────────────

SIZE_VALUES = ["tiny", "small", "medium", "large", "huge", "astronomical"]

# Default size per entity_type
ENTITY_SIZE_CLASSES: dict[str, str] = {
    "agent":          "medium",
    "animal":         "medium",
    "plant":          "medium",
    "vehicle":        "large",
    "robot":          "medium",
    "machine":        "large",
    "furniture":      "medium",
    "tool":           "small",
    "container":      "small",
    "electronic":     "small",
    "weapon":         "small",
    "sports_object":  "small",
    "structure":      "huge",
    "terrain":        "huge",
    "fluid":          "large",
    "projectile":     "tiny",
    "particle":       "tiny",
    "celestial_body": "astronomical",
    "non_physical":   "unknown",
}

# Keyword → size overrides
SIZE_KEYWORD_OVERRIDES: dict[str, str] = {
    "dust":       "tiny",   "particle":  "tiny",  "atom":     "tiny",
    "molecule":   "tiny",   "nanoparticle":"tiny","proton":   "tiny",
    "bullet":     "tiny",   "pellet":    "tiny",  "spark":    "tiny",
    "pebble":     "small",  "ball":      "small", "cup":      "small",
    "bottle":     "small",  "knife":     "small", "phone":    "small",
    "laptop":     "small",  "book":      "small",
    "chair":      "medium", "table":     "medium","car":      "large",
    "person":     "medium", "dog":       "medium","wolf":     "medium",
    "elephant":   "large",  "whale":     "large", "tree":     "large",
    "truck":      "large",  "ship":      "huge",  "building": "huge",
    "skyscraper": "huge",   "mountain":  "huge",  "ocean":    "huge",
    "planet":     "astronomical","star":  "astronomical","galaxy":"astronomical",
    "asteroid":   "large",  "comet":     "large",
    "micro":      "tiny",   "nano":      "tiny",  "mini":     "small",
    "giant":      "huge",   "massive":   "huge",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12  –  NEW: Affordance ontology
#
# What actions can an agent perform on / with this entity?
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_AFFORDANCES: dict[str, list[str]] = {
    "agent":          ["move", "interact", "communicate", "grasp", "push"],
    "animal":         ["move", "avoid", "observe", "interact"],
    "plant":          ["observe", "cut", "water", "uproot"],
    "vehicle":        ["drive", "transport", "collide", "park", "steer"],
    "robot":          ["move", "interact", "manipulate", "program", "charge"],
    "machine":        ["operate", "power_on", "power_off", "adjust", "maintain"],
    "furniture":      ["sit_on", "place_on", "move", "push", "support_objects"],
    "tool":           ["grasp", "use", "cut", "measure", "fasten"],
    "container":      ["open", "close", "fill", "empty", "carry", "pour"],
    "electronic":     ["power_on", "power_off", "interact", "charge", "configure"],
    "weapon":         ["aim", "fire", "reload", "carry", "deploy"],
    "sports_object":  ["throw", "catch", "hit", "bounce", "roll", "kick"],
    "structure":      ["enter", "exit", "lean_on", "climb", "pass_through"],
    "terrain":        ["walk_on", "drive_on", "dig", "traverse"],
    "fluid":          ["flow", "pour", "splash", "contain", "evaporate"],
    "projectile":     ["launch", "dodge", "intercept", "detonate"],
    "particle":       ["inhale", "disperse", "settle", "filter"],
    "celestial_body": ["observe", "orbit", "land_on"],
    "non_physical":   [],
}

# Subclass-level affordance overrides / extensions
SUBCLASS_AFFORDANCES: dict[str, list[str]] = {
    "car":           ["drive", "park", "accelerate", "brake", "steer", "collide"],
    "truck":         ["drive", "load", "unload", "tow", "collide"],
    "aircraft":      ["fly", "land", "take_off", "navigate"],
    "watercraft":    ["sail", "dock", "navigate", "capsize"],
    "ball":          ["throw", "bounce", "roll", "kick", "catch"],
    "cutting_tool":  ["cut", "slice", "carve", "grasp"],
    "bottle":        ["pour", "fill", "grasp", "seal", "open"],
    "building":      ["enter", "exit", "occupy", "demolish"],
    "human":         ["walk", "run", "jump", "grasp", "communicate", "interact"],
    "drone":         ["fly", "hover", "navigate", "film", "deliver"],
    "firearm":       ["aim", "fire", "reload", "holster"],
    "melee_weapon":  ["swing", "stab", "block", "parry"],
    "liquid":        ["flow", "pour", "evaporate", "freeze", "dissolve"],
    "gas":           ["expand", "compress", "diffuse", "ignite"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13  –  NEW: Interaction properties (expanded)
#
# Replaces and extends the original PROPERTY_POOL with a richer set.
# These properties are physics-engine hints beyond the base "properties" field.
# ─────────────────────────────────────────────────────────────────────────────

INTERACTION_PROPERTIES_POOL: list[str] = [
    "rigid", "soft", "elastic", "deformable", "rolling", "motorized",
    "sharp", "fragile", "flammable", "conductive", "magnetic",
    "floating", "slippery", "sticky", "transparent", "reflective",
    "heavy", "compressible", "porous", "abrasive",
]

# Default interaction properties per entity_type
ENTITY_INTERACTION_PROPERTIES: dict[str, list[str]] = {
    "agent":          ["soft", "deformable"],
    "animal":         ["soft", "deformable"],
    "plant":          ["soft", "deformable", "flammable"],
    "vehicle":        ["rigid", "heavy", "conductive"],
    "robot":          ["rigid", "conductive", "magnetic"],
    "machine":        ["rigid", "heavy", "conductive"],
    "furniture":      ["rigid", "heavy"],
    "tool":           ["rigid", "sharp"],
    "container":      ["rigid", "hollow"],
    "electronic":     ["rigid", "conductive", "fragile"],
    "weapon":         ["rigid", "sharp"],
    "sports_object":  ["elastic", "rolling"],
    "structure":      ["rigid", "heavy", "abrasive"],
    "terrain":        ["rigid", "abrasive", "porous"],
    "fluid":          ["soft", "slippery", "deformable"],
    "projectile":     ["rigid", "sharp"],
    "particle":       ["soft", "deformable", "abrasive"],
    "celestial_body": ["rigid", "heavy", "magnetic"],
    "non_physical":   [],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14  –  NEW: Scene role priors
#
# What role does this entity typically play in a simulated scene?
# ─────────────────────────────────────────────────────────────────────────────

SCENE_ROLES_POOL: list[str] = [
    "actor", "obstacle", "support", "medium", "collider",
    "projectile", "container", "terrain", "background",
]

ENTITY_SCENE_ROLES: dict[str, list[str]] = {
    "agent":          ["actor", "collider"],
    "animal":         ["actor", "collider"],
    "plant":          ["obstacle", "background"],
    "vehicle":        ["actor", "collider"],
    "robot":          ["actor", "collider", "manipulator"],
    "machine":        ["actor", "obstacle"],
    "furniture":      ["support", "obstacle"],
    "tool":           ["actor"],
    "container":      ["container", "support"],
    "electronic":     ["actor", "support"],
    "weapon":         ["actor", "collider"],
    "sports_object":  ["projectile", "collider"],
    "structure":      ["obstacle", "background", "support"],
    "terrain":        ["terrain", "obstacle"],
    "fluid":          ["medium", "terrain"],
    "projectile":     ["projectile", "collider"],
    "particle":       ["medium", "background"],
    "celestial_body": ["obstacle", "terrain", "background"],
    "non_physical":   [],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15  –  NEW: Alias dictionary
#
# Common synonyms and alternative names per entity_type (entity-level).
# Token-specific aliases are looked up at generation time.
# ─────────────────────────────────────────────────────────────────────────────

# Maps specific tokens (lowercased) to a list of alias strings
TOKEN_ALIASES: dict[str, list[str]] = {
    "car":          ["automobile", "vehicle", "auto", "motorcar"],
    "truck":        ["lorry", "HGV", "semi"],
    "motorcycle":   ["motorbike", "bike", "moto"],
    "bicycle":      ["bike", "cycle", "pushbike"],
    "airplane":     ["aircraft", "plane", "aeroplane", "airliner"],
    "helicopter":   ["chopper", "helo", "rotorcraft"],
    "boat":         ["vessel", "watercraft", "craft"],
    "ship":         ["vessel", "ocean liner", "seafaring vessel"],
    "robot":        ["bot", "automaton", "android"],
    "drone":        ["UAV", "unmanned aerial vehicle", "quadcopter"],
    "smartphone":   ["mobile phone", "cellphone", "handheld"],
    "laptop":       ["notebook", "portable computer"],
    "couch":        ["sofa", "settee", "divan"],
    "chair":        ["seat", "seating"],
    "knife":        ["blade", "cutter"],
    "hammer":       ["mallet", "maul"],
    "bottle":       ["flask", "vessel"],
    "backpack":     ["rucksack", "knapsack", "pack"],
    "gun":          ["firearm", "pistol", "handgun"],
    "rifle":        ["long gun", "carbine"],
    "person":       ["human", "individual", "man", "woman"],
    "human":        ["person", "Homo sapiens", "individual"],
    "dog":          ["canine", "hound", "puppy"],
    "cat":          ["feline", "kitty", "kitten"],
    "horse":        ["steed", "mare", "stallion", "equine"],
    "tree":         ["timber", "plant", "flora"],
    "water":        ["H2O", "aqua", "liquid"],
    "ball":         ["sphere", "orb"],
    "building":     ["structure", "edifice", "construction"],
    "road":         ["highway", "street", "path", "lane"],
    "rock":         ["stone", "boulder", "pebble"],
    "bullet":       ["round", "cartridge", "projectile"],
    "missile":      ["rocket", "munition", "guided weapon"],
    "dust":         ["fine particles", "particulate", "powder"],
    "star":         ["sun", "solar body", "stellar object"],
    "planet":       ["world", "orb", "terrestrial body"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16  –  Data record dataclass  (extended from original)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityRecord:
    """
    A single training example for the hierarchical entity classifier.

    Original fields
    ───────────────
    token               : surface form fed to the tokenizer
    entity_type         : leaf-level ontology class (direct prediction target)
    parent_class        : intermediate ontology node (used in hierarchical loss)
    root_class          : always "physical_entity" (or "non_physical")
    superclass          : coarse label grouping for loss weighting
    subclass            : fine-grained label within entity_type
    properties          : original inferred physical properties
    confidence          : [0, 1]  — 1.0 for unambiguous, <1.0 for ambiguous
    variant_of          : canonical seed form, or None if this IS the seed
    variant_type        : e.g. "plural", "colour_adj", "compound", "hard_example"

    NEW fields (v2 — all optional with sensible defaults)
    ──────────────────────────────────────────────────────
    material            : primary material composition (see MATERIALS list)
    phase               : thermodynamic phase  (solid / liquid / gas / plasma / granular)
    mobility            : kinematic class  (static / movable / self_propelled / flowing / flying)
    size_class          : coarse size  (tiny / small / medium / large / huge / astronomical)
    affordances         : list of action labels an agent can perform w/ this entity
    interaction_properties : expanded physical interaction flags
    scene_roles         : typical scene roles  (actor / obstacle / support / ...)
    aliases             : alternative names / synonyms for this token
    negative            : True if this is a non-physical negative example
    possible_classes    : alternative plausible entity_type labels (ambiguous examples)
    """
    # ── original fields ───────────────────────────────────────────────────────
    token:                  str
    entity_type:            str
    parent_class:           str
    root_class:             str                 = "physical_entity"
    superclass:             str                 = ""
    subclass:               str                 = ""
    properties:             list[str]           = field(default_factory=list)
    confidence:             float               = 1.0
    variant_of:             Optional[str]       = None
    variant_type:           Optional[str]       = None

    # ── NEW fields ────────────────────────────────────────────────────────────
    material:               str                 = "unknown"
    phase:                  str                 = "solid"
    mobility:               str                 = "movable"
    size_class:             str                 = "medium"
    affordances:            list[str]           = field(default_factory=list)
    interaction_properties: list[str]           = field(default_factory=list)
    scene_roles:            list[str]           = field(default_factory=list)
    aliases:                list[str]           = field(default_factory=list)
    negative:               bool                = False
    possible_classes:       Optional[list[str]] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17  –  Core generator class  (extended from original)
# ─────────────────────────────────────────────────────────────────────────────

class EntityDatasetGenerator:
    """
    Generates a large, balanced, deduplicated JSONL dataset for training a
    hierarchical entity ontology classifier.

    The generator is deliberately stateless between generate() calls so that
    it can be embedded in a larger pipeline or called repeatedly with
    different seeds.

    Parameters
    ──────────
    seed          : random seed for full reproducibility
    max_samples   : hard cap on output records (0 = unlimited)
    balance       : if True, undersample majority classes to match minority
    verbose       : print progress to stdout
    """

    def __init__(
        self,
        seed:        int  = 42,
        max_samples: int  = 0,
        balance:     bool = False,
        verbose:     bool = True,
    ) -> None:
        self.seed        = seed
        self.max_samples = max_samples
        self.balance     = balance
        self.verbose     = verbose
        random.seed(seed)

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self) -> list[EntityRecord]:
        """
        Run the full generation pipeline and return a list of EntityRecord.

        Pipeline:
          1. Seed records (one per vocabulary entry, including non_physical)
          2. Plural variants
          3. Colour-adjective variants
          4. Size-adjective variants
          5. Material-adjective variants
          6. State-adjective variants
          7. Domain-adjective variants
          8. Compound-noun variants
          9. Case variants  (upper / lower / camelCase / hyphenated)
         10. Hard examples
         11. Ambiguous examples
         12. Deduplication
         13. Optional balancing
        """
        t0 = time.time()
        records: list[EntityRecord] = []

        # Steps 1–9: vocabulary expansion
        records.extend(self._generate_seeds())
        records.extend(self._generate_plural_variants(records))
        records.extend(self._generate_adjective_variants(records, "colour"))
        records.extend(self._generate_adjective_variants(records, "size"))
        records.extend(self._generate_adjective_variants(records, "material"))
        records.extend(self._generate_adjective_variants(records, "state"))
        records.extend(self._generate_domain_adjective_variants(records))
        records.extend(self._generate_compound_variants(records))
        records.extend(self._generate_case_variants(records))

        # Steps 10–11: curated additions
        records.extend(self._add_hard_examples())
        records.extend(self._add_ambiguous_examples())

        # Step 12: deduplication (preserve first occurrence)
        records = self._deduplicate(records)

        # Step 13: optional class balancing
        if self.balance:
            records = self._balance_classes(records)

        # Step 14: optional cap
        if self.max_samples > 0:
            random.shuffle(records)
            records = records[:self.max_samples]

        # Final sort for deterministic output (token then entity_type)
        records.sort(key=lambda r: (r.token.lower(), r.entity_type))

        elapsed = time.time() - t0
        if self.verbose:
            self._print_stats(records, elapsed)

        return records

    def save(self, records: list[EntityRecord], path: str) -> None:
        """Write records to a JSONL file, creating parent directories if needed."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        if self.verbose:
            print(f"[DatasetGen] saved {len(records):,} records → {out}")

    # ── step 1: seed records ──────────────────────────────────────────────────

    def _generate_seeds(self) -> list[EntityRecord]:
        """Create one EntityRecord per entry in SEED_VOCAB (all classes)."""
        records: list[EntityRecord] = []
        for entity_type, tokens in SEED_VOCAB.items():
            is_negative = (entity_type == "non_physical")

            if is_negative:
                parent_class = "abstract"
                root_class   = "non_physical"
            else:
                parent_class, root_class = ONTOLOGY[entity_type]

            base_props = ENTITY_PROPERTIES.get(entity_type, [])

            for token in tokens:
                subclass, extra_props = self._pick_subclass(token, entity_type)
                props = _dedupe_list(base_props + extra_props)

                # Resolve new extended fields
                material   = self._resolve_material(token, entity_type)
                phase      = self._resolve_phase(token, entity_type)
                mobility   = self._resolve_mobility(token, entity_type)
                size_class = self._resolve_size(token, entity_type)
                affordances = self._resolve_affordances(token, entity_type, subclass)
                int_props   = self._resolve_interaction_properties(entity_type)
                scene_roles = ENTITY_SCENE_ROLES.get(entity_type, [])
                aliases     = TOKEN_ALIASES.get(token.lower(), [])

                records.append(EntityRecord(
                    token=                  token,
                    entity_type=            entity_type,
                    parent_class=           parent_class,
                    root_class=             root_class,
                    superclass=             entity_type,
                    subclass=               subclass,
                    properties=             props,
                    confidence=             1.0,
                    variant_of=             None,
                    variant_type=           None,
                    # new fields
                    material=               material,
                    phase=                  phase,
                    mobility=               mobility,
                    size_class=             size_class,
                    affordances=            affordances,
                    interaction_properties= int_props,
                    scene_roles=            scene_roles,
                    aliases=                aliases,
                    negative=               is_negative,
                    possible_classes=       None,
                ))
        return records

    # ── step 2: plural variants ───────────────────────────────────────────────

    def _generate_plural_variants(
        self, seeds: list[EntityRecord]
    ) -> list[EntityRecord]:
        """
        Generate English plural forms for seed tokens.

        Rules applied (in order):
          • ends in 'fe'   → 'ves'  (knife → knives)
          • ends in 'f'    → 'ves'  (leaf → leaves)
          • ends in 'us'   → 'i'    (radius → radii)
          • ends in 'is'   → 'es'   (axis → axes)
          • ends in vowel+'o' → 'es'
          • ends in 's','x','z','ch','sh' → 'es'
          • ends in consonant+'y' → 'ies'
          • otherwise → 's'
        """
        new_records: list[EntityRecord] = []
        seed_tokens = {r.token.lower() for r in seeds}

        for rec in seeds:
            if rec.variant_type is not None:
                continue  # only pluralise seeds
            if rec.negative:
                continue  # no plural of abstract nouns
            plural = _pluralise(rec.token)
            if plural.lower() == rec.token.lower():
                continue
            if plural.lower() in seed_tokens:
                continue
            new_records.append(EntityRecord(
                token=                  plural,
                entity_type=            rec.entity_type,
                parent_class=           rec.parent_class,
                root_class=             rec.root_class,
                superclass=             rec.superclass,
                subclass=               rec.subclass,
                properties=             rec.properties,
                confidence=             1.0,
                variant_of=             rec.token,
                variant_type=           "plural",
                # propagate new fields
                material=               rec.material,
                phase=                  rec.phase,
                mobility=               rec.mobility,
                size_class=             rec.size_class,
                affordances=            rec.affordances,
                interaction_properties= rec.interaction_properties,
                scene_roles=            rec.scene_roles,
                aliases=                rec.aliases,
                negative=               rec.negative,
                possible_classes=       rec.possible_classes,
            ))
        return new_records

    # ── steps 3–7: adjective variants ─────────────────────────────────────────

    def _generate_adjective_variants(
        self,
        records:  list[EntityRecord],
        adj_type: str,
    ) -> list[EntityRecord]:
        """
        Prepend adjectives of the given type to a random sample of seed tokens.

        adj_type is one of: 'colour', 'size', 'material', 'state'.
        We sample a subset of seeds and adjectives to avoid combinatorial
        explosion while still covering diverse surface forms.
        """
        adjectives = ADJECTIVES.get(adj_type, [])
        if not adjectives:
            return []

        seeds = [r for r in records if r.variant_type is None and not r.negative]
        new_records: list[EntityRecord] = []

        for rec in seeds:
            chosen_adjs = random.sample(adjectives, min(3, len(adjectives)))
            for adj in chosen_adjs:
                token = f"{adj} {rec.token}"
                new_records.append(self._make_variant(rec, token, f"{adj_type}_adj"))
        return new_records

    def _generate_domain_adjective_variants(
        self, records: list[EntityRecord]
    ) -> list[EntityRecord]:
        """Prepend domain-specific adjectives (from ADJECTIVES['domain'])."""
        domain_adjs: dict = ADJECTIVES.get("domain", {})
        seeds = [r for r in records if r.variant_type is None and not r.negative]
        new_records: list[EntityRecord] = []

        for rec in seeds:
            adjs = domain_adjs.get(rec.entity_type, [])
            if not adjs:
                continue
            for adj in random.sample(adjs, min(2, len(adjs))):
                token = f"{adj} {rec.token}"
                new_records.append(self._make_variant(rec, token, "domain_adj"))
        return new_records

    # ── step 8: compound variants ─────────────────────────────────────────────

    def _generate_compound_variants(
        self, records: list[EntityRecord]
    ) -> list[EntityRecord]:
        """Prepend compound modifiers (sports car, off-road truck, etc.)."""
        seeds = [r for r in records if r.variant_type is None and not r.negative]
        new_records: list[EntityRecord] = []

        for rec in seeds:
            mods = COMPOUND_MODIFIERS.get(rec.entity_type, [])
            if not mods:
                continue
            for mod in random.sample(mods, min(2, len(mods))):
                token = f"{mod} {rec.token}"
                new_records.append(self._make_variant(rec, token, "compound"))
        return new_records

    # ── step 9: case and typography variants ──────────────────────────────────

    def _generate_case_variants(
        self, records: list[EntityRecord]
    ) -> list[EntityRecord]:
        """
        Generate surface-form variants:
          • UPPERCASE  (TESLA)
          • lowercase  (tesla)
          • CamelCase  (Tesla)
          • hyphenated (sports-car)
        Only applied to single-word or two-word seed tokens.
        """
        seeds = [r for r in records if r.variant_type is None]
        new_records: list[EntityRecord] = []

        for rec in seeds:
            words = rec.token.split()

            up = rec.token.upper()
            if up != rec.token:
                new_records.append(self._make_variant(rec, up, "uppercase"))

            low = rec.token.lower()
            if low != rec.token:
                new_records.append(self._make_variant(rec, low, "lowercase"))

            if len(words) > 1:
                camel = "".join(w.capitalize() for w in words)
                new_records.append(self._make_variant(rec, camel, "camelcase"))

            if len(words) == 2:
                hyph = "-".join(words)
                new_records.append(self._make_variant(rec, hyph, "hyphenated"))

        return new_records

    # ── step 10: hard examples ────────────────────────────────────────────────

    def _add_hard_examples(self) -> list[EntityRecord]:
        """Inject the curated hard-example list with slightly reduced confidence."""
        records: list[EntityRecord] = []
        for token, entity_type in HARD_EXAMPLES:
            parent_class, root_class = ONTOLOGY[entity_type]
            subclass, extra_props = self._pick_subclass(token, entity_type)
            base_props = ENTITY_PROPERTIES.get(entity_type, [])
            props = _dedupe_list(base_props + extra_props)

            records.append(EntityRecord(
                token=                  token,
                entity_type=            entity_type,
                parent_class=           parent_class,
                root_class=             root_class,
                superclass=             entity_type,
                subclass=               subclass,
                properties=             props,
                confidence=             0.95,
                variant_of=             None,
                variant_type=           "hard_example",
                material=               self._resolve_material(token, entity_type),
                phase=                  self._resolve_phase(token, entity_type),
                mobility=               self._resolve_mobility(token, entity_type),
                size_class=             self._resolve_size(token, entity_type),
                affordances=            self._resolve_affordances(token, entity_type, subclass),
                interaction_properties= self._resolve_interaction_properties(entity_type),
                scene_roles=            ENTITY_SCENE_ROLES.get(entity_type, []),
                aliases=                TOKEN_ALIASES.get(token.lower(), []),
                negative=               False,
                possible_classes=       None,
            ))
        return records

    # ── step 11: ambiguous examples ───────────────────────────────────────────

    def _add_ambiguous_examples(self) -> list[EntityRecord]:
        """Inject the curated ambiguous-example list with reduced confidence."""
        records: list[EntityRecord] = []
        for ex in AMBIGUOUS_EXAMPLES:
            entity_type = ex["entity_type"]
            if entity_type == "non_physical":
                parent_class, root_class = "abstract", "non_physical"
            else:
                parent_class, root_class = ONTOLOGY[entity_type]

            subclass, extra_props = self._pick_subclass(ex["token"], entity_type)
            base_props = ENTITY_PROPERTIES.get(entity_type, [])
            props = _dedupe_list(base_props + extra_props)

            records.append(EntityRecord(
                token=                  ex["token"],
                entity_type=            entity_type,
                parent_class=           parent_class,
                root_class=             root_class,
                superclass=             entity_type,
                subclass=               subclass,
                properties=             props,
                confidence=             ex.get("confidence", 0.7),
                variant_of=             None,
                variant_type=           "ambiguous",
                material=               self._resolve_material(ex["token"], entity_type),
                phase=                  self._resolve_phase(ex["token"], entity_type),
                mobility=               self._resolve_mobility(ex["token"], entity_type),
                size_class=             self._resolve_size(ex["token"], entity_type),
                affordances=            self._resolve_affordances(ex["token"], entity_type, subclass),
                interaction_properties= self._resolve_interaction_properties(entity_type),
                scene_roles=            ENTITY_SCENE_ROLES.get(entity_type, []),
                aliases=                TOKEN_ALIASES.get(ex["token"].lower(), []),
                negative=               (entity_type == "non_physical"),
                possible_classes=       ex.get("possible_classes", None),
            ))
        return records

    # ── step 12: deduplication ────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(records: list[EntityRecord]) -> list[EntityRecord]:
        """
        Remove exact duplicate (token, entity_type) pairs, keeping the first
        occurrence (the highest-confidence / seed version).
        """
        seen: set[tuple[str, str]] = set()
        out:  list[EntityRecord]   = []
        for rec in records:
            key = (rec.token.lower(), rec.entity_type)
            if key not in seen:
                seen.add(key)
                out.append(rec)
        return out

    # ── step 13: class balancing ──────────────────────────────────────────────

    @staticmethod
    def _balance_classes(records: list[EntityRecord]) -> list[EntityRecord]:
        """
        Undersample majority classes so every entity_type has the same count
        as the smallest class.
        """
        from collections import defaultdict
        buckets: dict[str, list[EntityRecord]] = defaultdict(list)
        for rec in records:
            buckets[rec.entity_type].append(rec)

        min_count = min(len(v) for v in buckets.values())
        balanced:  list[EntityRecord] = []
        for recs in buckets.values():
            random.shuffle(recs)
            balanced.extend(recs[:min_count])
        return balanced

    # ── variant factory helper ────────────────────────────────────────────────

    @staticmethod
    def _make_variant(rec: EntityRecord, token: str, vtype: str) -> EntityRecord:
        """
        Create a variant EntityRecord from a seed, overriding token and
        variant_type while propagating all other fields (including new v2
        fields) unchanged.
        """
        return EntityRecord(
            token=                  token,
            entity_type=            rec.entity_type,
            parent_class=           rec.parent_class,
            root_class=             rec.root_class,
            superclass=             rec.superclass,
            subclass=               rec.subclass,
            properties=             rec.properties,
            confidence=             rec.confidence,
            variant_of=             rec.token,
            variant_type=           vtype,
            material=               rec.material,
            phase=                  rec.phase,
            mobility=               rec.mobility,
            size_class=             rec.size_class,
            affordances=            rec.affordances,
            interaction_properties= rec.interaction_properties,
            scene_roles=            rec.scene_roles,
            aliases=                rec.aliases,
            negative=               rec.negative,
            possible_classes=       rec.possible_classes,
        )

    # ── NEW: extended field resolvers ─────────────────────────────────────────

    @staticmethod
    def _resolve_material(token: str, entity_type: str) -> str:
        """Keyword-match → entity_type default fallback."""
        tok = token.lower()
        for kw, mat in MATERIAL_KEYWORD_OVERRIDES.items():
            if kw in tok:
                return mat
        return ENTITY_MATERIALS.get(entity_type, "unknown")

    @staticmethod
    def _resolve_phase(token: str, entity_type: str) -> str:
        """Keyword-match → entity_type default fallback."""
        tok = token.lower()
        for kw, phase in PHASE_KEYWORD_OVERRIDES.items():
            if kw in tok:
                return phase
        return ENTITY_PHASES.get(entity_type, "solid")

    @staticmethod
    def _resolve_mobility(token: str, entity_type: str) -> str:
        """Keyword-match → entity_type default fallback."""
        tok = token.lower()
        for kw, mob in MOBILITY_KEYWORD_OVERRIDES.items():
            if kw in tok:
                return mob
        return ENTITY_MOBILITY.get(entity_type, "movable")

    @staticmethod
    def _resolve_size(token: str, entity_type: str) -> str:
        """Keyword-match → entity_type default fallback."""
        tok = token.lower()
        for kw, sz in SIZE_KEYWORD_OVERRIDES.items():
            if kw in tok:
                return sz
        return ENTITY_SIZE_CLASSES.get(entity_type, "medium")

    @staticmethod
    def _resolve_affordances(
        token: str, entity_type: str, subclass: str
    ) -> list[str]:
        """
        Merge entity-level affordances with any subclass-specific overrides.
        Subclass affordances extend (not replace) the entity-type list.
        """
        base = list(ENTITY_AFFORDANCES.get(entity_type, []))
        sub  = list(SUBCLASS_AFFORDANCES.get(subclass, []))
        return _dedupe_list(sub + base)

    @staticmethod
    def _resolve_interaction_properties(entity_type: str) -> list[str]:
        """Return the canonical interaction properties for this entity_type."""
        return list(ENTITY_INTERACTION_PROPERTIES.get(entity_type, []))

    # ── original helpers (preserved verbatim) ─────────────────────────────────

    @staticmethod
    def _pick_subclass(token: str, entity_type: str) -> tuple[str, list[str]]:
        """
        Heuristically assign a subclass to a token by keyword matching.
        Falls back to the first subclass in SUBCLASS_MAP.
        """
        subcls_opts = SUBCLASS_MAP.get(entity_type, [])
        if not subcls_opts:
            return ("", [])

        tok_lower = token.lower()
        for subclass, extra_props in subcls_opts:
            subclass_words = subclass.replace("_", " ").split()
            if any(w in tok_lower for w in subclass_words):
                return (subclass, extra_props)

        return (subcls_opts[0][0], subcls_opts[0][1])

    @staticmethod
    def _print_stats(records: list[EntityRecord], elapsed: float) -> None:
        """Print a per-class breakdown to stdout."""
        from collections import Counter
        counts = Counter(r.entity_type for r in records)
        total  = len(records)
        print(f"\n[DatasetGen] generation complete  {total:,} records  "
              f"({elapsed:.1f}s)")
        print(f"{'entity_type':<22} {'count':>8}  {'%':>6}")
        print("─" * 42)
        for etype in sorted(counts):
            n = counts[etype]
            print(f"{etype:<22} {n:>8,}  {100*n/total:>5.1f}%")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18  –  Module-level helper functions  (original — preserved verbatim)
# ─────────────────────────────────────────────────────────────────────────────

def _dedupe_list(lst: list[str]) -> list[str]:
    """Return lst with duplicates removed, preserving order."""
    seen: set[str] = set()
    out:  list[str] = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _pluralise(word: str) -> str:
    """
    Generate an English plural form for a given token (or the last word of a
    multi-word token).  Handles common irregular patterns.
    """
    parts = word.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0] + " " + _pluralise(parts[1])

    w = word
    _IRREGULARS = {
        "person": "people", "man": "men", "woman": "women",
        "child": "children", "mouse": "mice", "goose": "geese",
        "tooth": "teeth", "foot": "feet", "ox": "oxen",
        "fish": "fish", "sheep": "sheep", "deer": "deer",
        "aircraft": "aircraft", "spacecraft": "spacecraft",
    }
    if w.lower() in _IRREGULARS:
        return _IRREGULARS[w.lower()]

    wl = w.lower()
    if wl.endswith("s") and not wl.endswith("ss"):
        return word
    if wl.endswith("fe"):
        return w[:-2] + "ves"
    if wl.endswith("f") and not wl.endswith("ff"):
        return w[:-1] + "ves"
    if wl.endswith("us") and len(wl) > 4:
        return w[:-2] + "i"
    if wl.endswith("is") and len(wl) > 3:
        return w[:-2] + "es"
    if wl.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if wl.endswith("y") and len(wl) > 1 and wl[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19  –  CLI entry point  (original — preserved verbatim)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a hierarchical entity classification dataset "
                    "for PhysWorldLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default="datasets/entity_classification.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int, default=0,
        help="Maximum number of output records (0 = unlimited).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--balance", "-b",
        action="store_true",
        help="Undersample majority classes to match minority class count.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for command-line execution."""
    args = _parse_args()

    gen = EntityDatasetGenerator(
        seed=        args.seed,
        max_samples= args.samples,
        balance=     args.balance,
        verbose=     not args.quiet,
    )

    records = gen.generate()
    gen.save(records, args.output)


if __name__ == "__main__":
    main()
