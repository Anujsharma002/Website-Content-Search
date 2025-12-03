import React, { useState } from "react";
import SearchBar from "./components/SearchBar";
import ResultCard from "./components/ResultCard";
import "./App.css";

function App() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  function dedupeFrontend(results) {
  const seen = new Set();
  const unique = [];

  for (const item of results) {
    const text = item.content.trim();
    if (!seen.has(text)) {
      seen.add(text);
      unique.push(item);
    }
  }

  return unique;
}

 async function handleSearch(url, query) {
  try {
    setLoading(true);

    const res = await fetch("http://localhost:8000/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, query }),
    });

    const data = await res.json();
    const uniqueResults = dedupeFrontend(data);

    setResults(uniqueResults);
  } catch (error) {
    console.error("Search failed:", error);
  } finally {
    setLoading(false);
  }
}




  return (
    <div className="app-container">
      <h1 className="title">Website Content Search</h1>
      <p className="subtitle">Search through website content with precision</p>

      <SearchBar onSearch={handleSearch} loading={loading}/>

      {results.length > 0 && <h2 className="results-title">Search Results</h2>}

      {results.map((item, index) => (
        <ResultCard key={index} item={item} loading={loading}/>
      ))}
    </div>
  );
}

export default App;
