"use client";

import type React from "react";
import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Skeleton } from "@/components/ui/skeleton";
import { Send, Bot, Sun, Moon, Mic, MicOff } from "lucide-react";
import { ChatMessage } from "@/components/chat-message";
import { useTheme } from "next-themes";
import { motion } from "framer-motion";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  data?: any[];
  chart?: any;
  timestamp: Date;
}

interface ApiResponse {
  response: string;
  data?: any[];
  chart?: any;
  chart_type?: string;
}

// Suggestion prompts
const suggestionPrompts = [
  "Show the trend of annual rainfall over years",
  "Compare groundwater levels between states",
  "Compare average stage of extraction between States",
  "Show category distribution",
];

// Mic visualizer
function MicVisualizer() {
  const bars = Array.from({ length: 5 });
  return (
    <div className="flex items-end gap-[3px] h-5">
      {bars.map((_, i) => (
        <motion.div
          key={i}
          className="w-[3px] bg-primary rounded-full"
          animate={{ height: ["20%", "100%", "40%", "70%", "30%"] }}
          transition={{
            duration: 1,
            repeat: Infinity,
            delay: i * 0.15,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [latitude, setLatitude] = useState<number | null>(null);
  const [longitude, setLongitude] = useState<number | null>(null);
  const [apiUrl] = useState(
    process.env.NEXT_PUBLIC_API_URL || "https://neersetu.onrender.com"
  );
  const [mounted, setMounted] = useState(false);

  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const inputValueRef = useRef<string>("");

  const { theme, setTheme, resolvedTheme } = useTheme();

  // Auto scroll
  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  };

  useEffect(() => {
    setMounted(true);
  }, []);

  // Geolocation
  useEffect(() => {
    if (typeof window !== "undefined" && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setLatitude(pos.coords.latitude);
          setLongitude(pos.coords.longitude);
        },
        () => {},
        { enableHighAccuracy: false, timeout: 5000, maximumAge: 300000 }
      );
    }
  }, []);

  useEffect(() => {
    inputValueRef.current = input;
  }, [input]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Send message
  const sendMessage = async (messageText?: string) => {
    const textToSend = messageText || input;
    if (!textToSend.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: textToSend,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${apiUrl}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: textToSend,
          latitude: latitude ?? undefined,
          longitude: longitude ?? undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response,
        data: data.data,
        chart: data.chart,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "Sorry, I encountered an error while processing your request. Please make sure the backend API is running and accessible.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Key press send
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleSuggestionClick = (prompt: string) => {
    sendMessage(prompt);
  };

  // Theme toggle
  const toggleTheme = () => {
    const currentTheme = resolvedTheme || theme;
    setTheme(currentTheme === "dark" ? "light" : "dark");
  };

  // Voice handling
  const startListening = () => {
    if (typeof window === "undefined") return;
    // @ts-ignore
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("SpeechRecognition is not supported in this browser.");
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.interimResults = true;
      recognition.continuous = true;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => {
        setIsListening(true);
      };
      recognition.onerror = () => {
        setIsListening(false);
      };
      recognition.onend = () => {
        if (isListening) {
          try {
            recognition.start();
          } catch {}
        } else {
          setIsListening(false);
        }
      };
      recognition.onresult = (event: any) => {
        let finalTranscript = "";
        let interimTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const res = event.results[i];
          if (res.isFinal) {
            finalTranscript += res[0]?.transcript ?? "";
          } else {
            interimTranscript += res[0]?.transcript ?? "";
          }
        }
        const combined = `${finalTranscript} ${interimTranscript}`.trim();
        if (combined) {
          setInput(combined);
        }
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch (e) {
      console.error("Failed to start speech recognition", e);
      setIsListening(false);
    }
  };

  const stopListening = () => {
    try {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    } catch {}
    setIsListening(false);

    // Auto-send when mic stops
    const latestText = inputValueRef.current;
    if (latestText && latestText.trim().length > 0 && !isLoading) {
      sendMessage(latestText);
    }
  };

  useEffect(() => {
    return () => {
      try {
        if (recognitionRef.current) {
          recognitionRef.current.stop();
        }
      } catch {}
    };
  }, []);

  // Skeleton before mount
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
    );
  }

  // Main UI
  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
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
            {(resolvedTheme || theme) === "dark" ? (
              <Sun className="w-4 h-4" />
            ) : (
              <Moon className="w-4 h-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Chat content */}
      <div className="flex-1 flex flex-col">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center p-8 max-w-2xl mx-auto w-full">
            <div className="text-center mb-12">
              <h1 className="text-2xl font-semibold text-foreground mb-2">
                Hello there!
              </h1>
              <p className="text-muted-foreground">
                I'm NeerSetu, your groundwater level assistant. How can I help
                you today?
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

        {/* Input area */}
        <div className="p-4 border-t">
          <div className="max-w-2xl mx-auto">
            <div className="relative flex items-center">
              <Input
                placeholder="Send a message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading}
                className="pr-28 py-3 rounded-full border-border"
              />

              {/* Mic + Send */}
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                {isListening && (
                  <div className="flex items-center gap-2 px-2 py-1 bg-muted rounded-lg">
                    <MicVisualizer />
                    <span className="text-xs text-muted-foreground max-w-[120px] truncate">
                      {input || "Listening..."}
                    </span>
                  </div>
                )}

                <Button
                  onClick={() =>
                    isListening ? stopListening() : startListening()
                  }
                  disabled={isLoading}
                  size="icon"
                  className={`w-8 h-8 rounded-full transition-colors ${
                    isListening ? "bg-red-500 hover:bg-red-600 text-white" : ""
                  }`}
                  aria-label={
                    isListening ? "Close mic & send" : "Start voice input"
                  }
                >
                  {isListening ? (
                    <MicOff className="w-4 h-4" />
                  ) : (
                    <Mic className="w-4 h-4" />
                  )}
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
  );
}
