"use client"

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, SquareArrowUpRight } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

function Result({ score }: { score?: { negScore: number, posScore: number, tokens: string[] }}) {
	/**
	 * negScore: float, 0-1, inclusive
	 * posScore: float, 0-1, inclusive
	 * tokens: string[], string of tokens
	 */
	const isPos = (score?.posScore ?? 0) >= 0.5

	return (
		<div className="flex flex-col grow text-zinc-200">
			<div className="flex flex-col" style={{
				color: score ? (isPos ? `var(--green)` : `var(--red)`) : "currentcolor"
			}}>
				<div className="relative w-8 h-8 mt-8 mb-6 md:w-32 md:h-32 md:mt-16 md:mb-12 self-center origin-center bg-current transition-colors" style={{
					animationName: "rotate",
					animationTimingFunction: "linear",
					animationIterationCount: "infinite",
					animationDuration: "20s"
				}}>
					<div className="absolute w-full h-full origin-center" style={{
						backgroundColor: "#fff",
						animationName: "shrink",
						animationTimingFunction: "cubic-bezier(.4, 0, .2, 0)",
						animationIterationCount: "infinite",
						animationDirection: "alternate",
						animationDuration: "1s"
					}}>
					</div>
				</div>
				<p className="font-bold self-center">{score ? (isPos ? `Positive` : `Negative`) : `Press send!`}</p>
			</div>
			<div className="flex flex-col grow gap-4 -ml-4 mt-12 overflow-x-clip">
				<div className="flex flex-row gap-2 transition-colors" style={{
					color: score ? `var(--red)` : `currentcolor` // inherit from parent
				}}>
					<div className="w-[4px] rounded-r-sm bg-current">
					</div>
					<div className="flex flex-col gap-2 transition-transform" style={{
						transform: score ? "" : `translateX(calc(-100% - 8px)`
					}}>
						<p className="font-bold text-4xl md:text-6xl">{`${((score?.negScore ?? 0) *100).toFixed(2)}`}<span className="text-2xl">%</span></p>
						<p className="font-bold">Negative</p>
					</div>
				</div>
				<div className="flex flex-row gap-2 transition-colors" style={{
					color: score ? `var(--green)` : `currentcolor` // inherit from parent
				}}>
					<div className="w-[4px] rounded-r-sm bg-current">
					</div>
					<div className="flex flex-col gap-2 transition-transform" style={{
						transform: score ? "" : `translateX(calc(-100% - 8px)`
					}}>
						<p className="font-bold text-4xl md:text-6xl">{`${((score?.posScore ?? 0) *100).toFixed(2)}`}<span className="text-2xl">%</span></p>
						<p className="font-bold">Positive</p>
					</div>
				</div>
			</div>
			{
				score &&
				<div className="flex flex-col gap-2 my-8 text-black">
					<p className="font-bold">Tokens</p>
					<p className="font-mono">{score.tokens.join(", ")}</p>
				</div>
			}
		</div>
	)
}

export default function Home() {
	const [score, setScore] = useState<{ negScore: number, posScore: number, tokens: string[] }|undefined>()

	const [inputText, setInputText] = useState("")
	const [sendDebounce, setSendDebounce] = useState(true)

	const exampleTexts = useMemo(() => [
		"best italian restaurant with authentic food",
		"the ice cream was the best",
		"the staff was not very friendly"
	], [])
	const [exampleText, setExampleText] = useState(exampleTexts[0])
	useEffect(() => {
		setExampleText(exampleTexts[Math.floor(Math.random() *exampleTexts.length)])
	}, [exampleTexts])

	return (
		<div className="w-full h-full p-4 flex flex-row justify-center">
			<div className="w-full md:w-[512px] h-full flex flex-col">
				<div className="w-full flex flex-col gap-2">
					<Label htmlFor="base">Text</Label>
					<Input disabled={!sendDebounce} id="base" type="text" placeholder="amazing!" value={inputText} onChange={(e) => setInputText(e.target.value)} />
					<button disabled={!sendDebounce} className="px-2 rounded bg-zinc-200 text-zinc-400 inline-flex flex-row items-center gap-1 text-left hover:text-zinc-700 focus:text-zinc-700 transition-colors" suppressHydrationWarning onClick={() => {
						if (!sendDebounce) {
							// currently sending, ignore
							return
						}
						setInputText(exampleText)
					}}>
						{exampleText}
						<SquareArrowUpRight size={16} strokeWidth={1.5} />
					</button>
					<Button disabled={!sendDebounce} className="self-end" onClick={async () => {
						if (inputText.trim().length === 0) {
							// empty
							toast.error("Input is empty")
							return
						}
						if (!sendDebounce) {
							// debounce triggered
							return
						}
						setSendDebounce(false) // set debounce

						// fetch from server
						try {
							const fetchPromise = fetch("/api/sentiment", {
								method: "POST",
								headers: {
									"Content-Type": "application/json"
								},
								body: JSON.stringify({
									text: inputText
								})
							}).then(async r => {
								const data = await r.json()
								if ("error" in data) {
									throw new Error(data.error)
								}

								return data as { neg: number, pos: number, sentiment: "pos"|"neg", tokens: string[] }
							})
							toast.promise(fetchPromise, { // create toast
								loading: "Classifying...",
								success: (data) => {
									return `Finished classifying! ${data.sentiment}`
								},
								error: "Failed to classify."
							})

							// set states
							const data = await fetchPromise
							if (data) {
								setScore({ negScore: data.neg, posScore: data.pos, tokens: data.tokens })
							} else {

							}
						} catch (err) {
							// @ts-expect-error error requires any type which fails lint
							toast.error(err.message)
						} finally {
							// reset debounce
							setSendDebounce(true)
						}
					}}>
					{
						sendDebounce ? (`Send`) : (
							<>
								<span>Please wait</span>
								<Loader2 className="animate-spin" />
							</>
						)
					}
					</Button>
				</div>
				<Result score={score} />
			</div>
		</div>
	)
}