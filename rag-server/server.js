// server.js
const express = require("express");
const { spawn } = require("child_process");
const axios = require("axios");

const app = express();
app.use(express.json());

app.post("/rag", async (req, res) => {
  const { question, top_k = 5 } = req.body;

  const py = spawn("python", ["embed.py", question, top_k.toString()]);

  py.on("error", (err) => {
    console.error("Failed to start subprocess:", err);
  });

  let data = "";

  py.stdout.on("data", (chunk) => {
    data += chunk.toString();
  });

  py.stderr.on("data", (err) => console.error("Python error:", err.toString()));

  py.on("close", async () => {
    try {
      const contextChunks = JSON.parse(data);
      const context = contextChunks.join("\n\n");

      const response = await axios.post("http://localhost:11434/api/generate", {
        model: "deepseek-r1:1.5b",
        prompt: `You are a factual assistant. Only use the provided context. Do not show reasoning steps, internal thoughts, or anything before the answer. Just give a direct, final answer.\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`,
        stream: false,
      });

      res.json({ answer: response.data.response });
    } catch (err) {
      console.error("Error:", err);
      res.status(500).json({ error: "Something went wrong." });
    }
  });
});

app.listen(3000, () => {
  console.log("🔊 RAG backend running on http://localhost:3000");
});
