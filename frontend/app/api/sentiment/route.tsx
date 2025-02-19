import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
	try {
		console.log("req.body", req.body)
		let { text } = await req.json();
		if (!text) {
			return NextResponse.json({ error: "Requires .text attribute in request body" }, { status: 400 });
		}

		// trimtext
		text = text.trim()

		// fetch response from internel server (Flask server running the classification)
		const response = await fetch("http://localhost:5000/api/sentiment", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ text }),
		});

		// forward error message too
		console.log("RESPONSE", response)
		if (!response.ok) {
			return NextResponse.json({ error: "Forward error" }, { status: response.status });
		}

		// return payload
		const data = await response.json();
		return NextResponse.json(data);
	} catch (error) {
		console.log("ERROR", error)
		return NextResponse.json({ error: "Server failed to process your request" }, { status: 500 });
	}
}
