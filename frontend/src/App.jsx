
import React, { useState } from "react";
import axios from "axios";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setAnswer("");

    try {
      const response = await axios.post("http://localhost:8000/generate", { input: question });
      setAnswer(response.data.answer);
    } catch (err) {
      setError("Failed to fetch answer from backend.");
    }
    setLoading(false);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#f5f7fa",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "1rem",
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
      }}
    >
      <div
        style={{
          backgroundColor: "#fff",
          padding: "2rem",
          borderRadius: "12px",
          boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
          maxWidth: 600,
          width: "100%",
        }}
      >
        <h1 style={{ textAlign: "center", color: "#222", marginBottom: "1.5rem" }}>
          Ask Your Question
        </h1>

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <input
            type="text"
            placeholder="Enter your question here..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            required
            style={{
              padding: "0.75rem 1rem",
              fontSize: "1.1rem",
              borderRadius: "8px",
              border: "1px solid #ccc",
              transition: "border-color 0.3s",
            }}
            onFocus={(e) => (e.target.style.borderColor = "#007bff")}
            onBlur={(e) => (e.target.style.borderColor = "#ccc")}
          />

          <button
            type="submit"
            disabled={loading}
            style={{
              padding: "0.75rem",
              fontSize: "1.1rem",
              backgroundColor: loading ? "#6c757d" : "#007bff",
              color: "#fff",
              border: "none",
              borderRadius: "8px",
              cursor: loading ? "not-allowed" : "pointer",
              transition: "background-color 0.3s",
            }}
            onMouseEnter={(e) => {
              if (!loading) e.target.style.backgroundColor = "#0056b3";
            }}
            onMouseLeave={(e) => {
              if (!loading) e.target.style.backgroundColor = "#007bff";
            }}
          >
            {loading ? "Loading..." : "Ask"}
          </button>
        </form>

        {error && (
          <p
            style={{
              marginTop: "1rem",
              color: "#dc3545",
              fontWeight: "600",
              textAlign: "center",
            }}
          >
            {error}
          </p>
        )}

        {answer && (
          <div
            style={{
              marginTop: "2rem",
              backgroundColor: "#e9f7ef",
              padding: "1rem 1.5rem",
              borderRadius: "10px",
              color: "#155724",
              boxShadow: "0 2px 10px rgba(21, 87, 36, 0.15)",
              fontSize: "1.05rem",
              lineHeight: "1.5",
              whiteSpace: "pre-wrap",
            }}
          >
            <h2 style={{ marginBottom: "0.5rem", fontWeight: "700" }}>Answer:</h2>
            <p>{answer}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
