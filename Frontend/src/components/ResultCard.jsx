import React, { useState } from "react";
import "./ResultCard.css";

function ResultCard({ item }) {
  const [showHTML, setShowHTML] = useState(false);

  return (
    <div className="result-card">
      <div className="result-header">
        <div className="result-title">{item.content.slice(0, 120)}...</div>
        <div className="score-badge">{Math.round(item.score * 100)}% match</div>
      </div>

      <p className="result-path">Path: /home</p>

      <button
        className="toggle-btn"
        onClick={() => setShowHTML(!showHTML)}
      >
        {showHTML ? "▼ Hide HTML" : "▲ View HTML"}
      </button>

      {showHTML && (
        <pre className="html-preview">
{item.html_content}
        </pre>
      )}
    </div>
  );
}

export default ResultCard;
