import React, { useState } from "react";
import "./SearchBar.css";

function SearchBar({ onSearch, loading }) {
  const [url, setUrl] = useState("");
  const [query, setQuery] = useState("");
  const [load,setload] = useState(false)
  function handleSubmit(e) {
    e.preventDefault();
    if (!loading) {
      onSearch(url, query);
    }
  }

  return (
    <form className="search-wrapper" onSubmit={handleSubmit}>
      <div className="input-box">
        <span className="input-icon">🌐</span>
        <input
          type="text"
          placeholder="https://example.com/"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={loading}
        />
      </div>

      <div className="input-box">
        <span className="input-icon">🔍</span>
        <input
          type="text"
          placeholder="AI"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
        />

        <button
          type="submit"
          className="search-btn"
          disabled={loading}
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
    </form>
  );
}

export default SearchBar;
