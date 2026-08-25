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
    throw new Error("Generation failed");
  }

  const result = await response.json();

  // Once the USD is generated, immediately open it in Omniverse.
  if (result.usd_path) {
    const omniverseResponse = await fetch(`${API_URL}/omniverse/show`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        usd_path: result.usd_path,
      }),
    });

    if (!omniverseResponse.ok) {
      throw new Error("USD generated, but failed to open in Omniverse");
    }

    result.omniverse = await omniverseResponse.json();
  }

  return result;
}
