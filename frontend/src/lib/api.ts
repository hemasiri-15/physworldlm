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

  return response.json();
}
