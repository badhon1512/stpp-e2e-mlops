import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const backendUrl = getBackendUrl();

  if (!backendUrl) {
    return NextResponse.json(
      { detail: "BACKEND_API_URL is not configured." },
      { status: 500 }
    );
  }

  let payload: unknown;

  try {
    payload = await request.json();
  } catch {
    return NextResponse.json(
      { detail: "Request body must be valid JSON." },
      { status: 400 }
    );
  }

  if (!isPredictionPayload(payload)) {
    return NextResponse.json(
      { detail: "subject and body are required." },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(`${backendUrl}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload),
      cache: "no-store"
    });

    const data = await response.json().catch(() => ({
      detail: "Backend returned a non-JSON response."
    }));

    if (!response.ok) {
      return NextResponse.json(
        {
          detail: getErrorMessage(data, "Prediction request failed."),
          backend_status: response.status
        },
        { status: response.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch {
    return NextResponse.json(
      { detail: "Prediction service is unavailable." },
      { status: 502 }
    );
  }
}

function isPredictionPayload(
  payload: unknown
): payload is { subject: string; body: string } {
  if (!payload || typeof payload !== "object") {
    return false;
  }

  const candidate = payload as { subject?: unknown; body?: unknown };
  return (
    typeof candidate.subject === "string" &&
    candidate.subject.trim().length >= 3 &&
    typeof candidate.body === "string" &&
    candidate.body.trim().length >= 5
  );
}

function getErrorMessage(data: unknown, fallback: string) {
  if (!data || typeof data !== "object") {
    return fallback;
  }

  const candidate = data as { detail?: unknown; error?: unknown };
  if (typeof candidate.detail === "string") {
    return candidate.detail;
  }
  if (typeof candidate.error === "string") {
    return candidate.error;
  }
  return fallback;
}

function getBackendUrl() {
  const backendUrl = process.env.BACKEND_API_URL?.trim();
  return backendUrl?.replace(/\/$/, "") || null;
}
