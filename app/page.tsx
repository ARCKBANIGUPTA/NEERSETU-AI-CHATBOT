"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Skeleton } from "@/components/ui/skeleton"
import { Send, Bot, Sun, Moon, Mic, MicOff } from "lucide-react"
import { ChatMessage } from "@/components/chat-message"
import { useTheme } from "next-themes"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  data?: any[]
  chart?: any
  timestamp: Date
}

interface ApiResponse {
  response: string
  data?: any[]
  chart?: any
  chart_type?: string
}

const suggestionPrompts = [
  "What is the current groundwater level in my area?",
  "Show me groundwater trends over the past year",
  "Analyze groundwater quality parameters for my location",
  "What factors affect groundwater depletion?",
]

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [apiUrl, setApiUrl] = useState(
    process.env.NEXT_PUBLIC_API_URL || "https://neersetu.onrender.com"
  )
  const [mounted, setMounted] = useState(false)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<any>(null)
  const silenceTimeoutRef = useRef<number | null>(null)
  const inputValueRef = useRef<string>("")
  const { theme, setTheme, resolvedTheme } = useTheme()

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector("[data-radix-scroll-area-viewport]")
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight
      }
    }
  }

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    inputValueRef.current = input
  }, [input])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async (messageText?: string) => {
    const textToSend = messageText || input
    if (!textToSend.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: textToSend,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch(`${apiUrl}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: textToSend }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: ApiResponse = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response,
        data: data.data,
        chart: data.chart,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error("Error sending message:", error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "Sorry, I encountered an error while processing your request. Please make sure the backend API is running and accessible.",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleSuggestionClick = (prompt: string) => {
    sendMessage(prompt)
  }

  const toggleTheme = () => {
    const currentTheme = resolvedTheme || theme
    setTheme(currentTheme === "dark" ? "light" : "dark")
  }

  // Voice input via Web Speech API (where available)
  const SILENCE_TIMEOUT_MS = 2000

  const clearSilenceTimer = () => {
    if (silenceTimeoutRef.current != null) {
      window.clearTimeout(silenceTimeoutRef.current)
      silenceTimeoutRef.current = null
    }
  }

  const startSilenceTimer = () => {
    clearSilenceTimer()
    silenceTimeoutRef.current = window.setTimeout(() => {
      // Auto stop and send if we have text
      stopListening()
      const latestText = inputValueRef.current
      if (latestText && latestText.trim().length > 0 && !isLoading) {
        sendMessage(latestText)
      }
    }, SILENCE_TIMEOUT_MS)
  }

  const startListening = () => {
    if (typeof window === "undefined") return
    // @ts-ignore - vendor prefixed API on some browsers
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      console.warn("SpeechRecognition is not supported in this browser.")
      return
    }

    try {
      const recognition = new SpeechRecognition()
      recognition.lang = "en-US"
      recognition.interimResults = true
      recognition.continuous = true
      recognition.maxAlternatives = 1
      recognition.onstart = () => {
        setIsListening(true)
        startSilenceTimer()
      }
      recognition.onerror = () => {
        setIsListening(false)
        clearSilenceTimer()
      }
      recognition.onend = () => {
        // If user is still in listening mode, restart recognition to keep streaming
        if (isListening) {
          try { recognition.start() } catch {}
        } else {
          setIsListening(false)
        }
        clearSilenceTimer()
      }
      recognition.onresult = (event: any) => {
        let finalTranscript = ""
        let interimTranscript = ""
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const res = event.results[i]
          if (res.isFinal) {
            finalTranscript += res[0]?.transcript ?? ""
          } else {
            interimTranscript += res[0]?.transcript ?? ""
          }
        }
        const combined = `${finalTranscript} ${interimTranscript}`.trim()
        if (combined) {
          setInput(combined)
        }
        startSilenceTimer()
      }
      recognitionRef.current = recognition
      recognition.start()
    } catch (e) {
      console.error("Failed to start speech recognition", e)
      setIsListening(false)
    }
  }

  const stopListening = () => {
    try {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
    } catch (e) {
      // no-op
    } finally {
      setIsListening(false)
      clearSilenceTimer()
    }
  }

  useEffect(() => {
    return () => {
      // cleanup on unmount
      try {
        if (recognitionRef.current) {
          recognitionRef.current.stop()
        }
      } catch (e) {
        // ignore
      }
    }
  }, [])

  if (!mounted) {
    return (
      <div className="flex flex-col h-screen bg-background">
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-foreground rounded-sm"></div>
            <span className="text-sm font-medium">NeerSetu</span>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" disabled>
              <Moon className="w-4 h-4" />
            </Button>
          </div>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <Skeleton className="h-8 w-32" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-foreground rounded-sm"></div>
          <span className="text-sm font-medium">NeerSetu</span>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="hover:bg-accent"
            aria-label="Toggle theme"
          >
            {(resolvedTheme || theme) === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </Button>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center p-8 max-w-2xl mx-auto w-full">
            <div className="text-center mb-12">
              <h1 className="text-2xl font-semibold text-foreground mb-2">Hello there!</h1>
              <p className="text-muted-foreground">
                I'm NeerSetu, your groundwater level assistant. How can I help you today?
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full mb-12">
              {suggestionPrompts.map((prompt, index) => (
                <Button
                  key={index}
                  variant="outline"
                  className="h-auto p-4 text-left justify-start whitespace-normal bg-transparent"
                  onClick={() => handleSuggestionClick(prompt)}
                >
                  {prompt}
                </Button>
              ))}
            </div>
          </div>
        ) : (
          <ScrollArea ref={scrollAreaRef} className="flex-1 p-4">
            <div className="space-y-4 max-w-4xl mx-auto">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}

              {isLoading && (
                <div className="flex items-start gap-3">
                  <Avatar className="w-8 h-8">
                    <AvatarFallback>
                      <Bot className="w-4 h-4" />
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        )}

        <div className="p-4 border-t">
          <div className="max-w-2xl mx-auto">
            <div className="relative">
              <Input
                placeholder="Send a message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading}
                className="pr-12 py-3 rounded-full border-border"
              />
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                <Button
                  onClick={() => (isListening ? stopListening() : startListening())}
                  disabled={isLoading}
                  size="icon"
                  className="w-8 h-8 rounded-full"
                  aria-label={isListening ? "Stop voice input" : "Start voice input"}
                >
                  {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                </Button>
                <Button
                  onClick={() => sendMessage()}
                  disabled={isLoading || !input.trim()}
                  size="icon"
                  className="w-8 h-8 rounded-full"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
