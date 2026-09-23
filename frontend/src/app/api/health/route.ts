import { NextResponse } from "next/server";

export async function GET() {
  try {
    const response = await fetch(
      "http://127.0.0.1:8000/health",
      {
        cache: "no-store",
      }
    );

    const data = await response.json();

    return NextResponse.json(data, {
      status: response.status,
    });
  } catch {
    return NextResponse.json(
      {
        status: "offline",
        device: "unknown",
        threshold: 0.5,
      },
      {
        status: 503,
      }
    );
  }
}