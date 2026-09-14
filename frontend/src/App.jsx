import { useState } from "react"
import axios from "axios"
import ReactMarkdown from "react-markdown"

function App() {
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  const askQuestion = async () => {
    if (!question.trim() || loading) {
      return
    }

    const userMessage = {
      role: "user",
      content: question
    }

    setMessages((previousMessages) => [
      ...previousMessages,
      userMessage
    ])

    setQuestion("")
    setLoading(true)

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/chat",
        {
          question: question,
          history: messages
        }
      )

      const assistantMessage = {
        role: "assistant",
        content: response.data.answer,
        sources: response.data.sources
      }

      setMessages((previousMessages) => [
        ...previousMessages,
        assistantMessage
      ])

    } catch (error) {
      const errorMessage = {
        role: "assistant",
        content: "Sorry, something went wrong. Please try again."
      }

      setMessages((previousMessages) => [
        ...previousMessages,
        errorMessage
      ])

    } finally {
      setLoading(false)
    }
  }

  return (
    
  <div className="app">
    <header className="header">
      <h1>MediLens</h1>
      <p>Clear medical knowledge, grounded in trusted sources.</p>
    </header>

    <main className="chat-container">
      {messages.length === 0 && (
        <div className="welcome">
          <h2>How can I help?</h2>
          <p>
            Ask a medical information question and I'll answer
            using the trusted sources in my knowledge base.
          </p>
        </div>
      )}

      <div className="messages">
        {messages.map((message, index) => (
          <div
            className={`message ${
              message.role === "user"
                ? "user-message"
                : "assistant-message"
            }`}
            key={index}
          >
            <div className="message-label">
  {message.role === "user" ? (
    "You"
  ) : (
    <span className="assistant-label">
      <span className="assistant-avatar">M</span>
      <span>MediLens</span>
      <span className="status-dot"></span>
    </span>
  )}
</div>

            <div className="message-content">
              <ReactMarkdown>
              {message.content}
              </ReactMarkdown>
            </div>

            {message.sources && (
              <div className="sources">
                <strong>Sources</strong>

                {message.sources.map((source, sourceIndex) => (
                  <div className="source-item" key={sourceIndex}>
                    <span>{source.source}</span>
                    <span>Page {source.page}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="message assistant-message">
            <div className="message-label">
  <span className="assistant-label">
    <span className="assistant-avatar">M</span>
    <span>MediLens</span>
    <span className="status-dot"></span>
  </span>
</div>
            <div className="thinking">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
      </div>
    </main>

    <div className="input-area">
      <input
        type="text"
        placeholder="Ask a medical question..."
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            askQuestion()
          }
        }}
      />

      <button
        onClick={askQuestion}
        disabled={loading || !question.trim()}
      >
        {loading ? "..." : "Ask"}
      </button>
    </div>

    <p className="disclaimer">
      MediLens provides general medical information, not diagnosis or treatment.
    </p>
  </div>

  )
}

export default App