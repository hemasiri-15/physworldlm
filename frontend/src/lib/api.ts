const API_URL = "http://localhost:8000";

export async function generateWorld(prompt: string) {
  const response = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prompt }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Generation failed");
  }

  return response.json();
}

export async function connectOmniverse() {
  const response = await fetch(`${API_URL}/omniverse/connect`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Omniverse connection failed");
  }

  return response.json();
}

export async function getOmniverseStatus() {
  const response = await fetch(`${API_URL}/omniverse/status`);

  if (!response.ok) {
    throw new Error("Failed to get Omniverse status");
  }

  return response.json();
}
