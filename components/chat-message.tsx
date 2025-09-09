import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Card } from "@/components/ui/card"
import { Bot, User } from "lucide-react"
import { cn } from "@/lib/utils"
import { ChartVisualization } from "./chart-visualization"
import { DataTable } from "./data-table"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  data?: any
  chart?: any
  timestamp: Date
}

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user"

  return (
    <div className={cn("flex items-start gap-3", isUser && "flex-row-reverse")}>
      <Avatar className="w-8 h-8">
        <AvatarFallback className={cn(isUser ? "bg-primary text-primary-foreground" : "bg-muted")}>
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </AvatarFallback>
      </Avatar>

      <Card className={cn("p-3 max-w-[80%]", isUser ? "bg-primary text-primary-foreground" : "bg-muted")}>
        <div className="space-y-3">
          {message.content && <div className="text-sm whitespace-pre-wrap">{message.content}</div>}

          {message.chart && <ChartVisualization chart={message.chart} />}

          {message.data && Array.isArray(message.data) && message.data.length > 0 && <DataTable data={message.data} />}
        </div>
      </Card>
    </div>
  )
}
