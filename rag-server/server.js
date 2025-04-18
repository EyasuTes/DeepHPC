// server.js
const express = require("express");
const { spawn } = require("child_process");
const axios = require("axios");

const app = express();
app.use(express.json());

app.post("/rag", async (req, res) => {
  const { question } = req.body;

const py = spawn("python", ["embed.py", question]);

py.on("error", (err) => {
  console.error("Failed to start subprocess:", err);
});

  let data = "";

  py.stdout.on("data", (chunk) => { data += chunk.toString(); });
  py.stderr.on("data", (err) => console.error("Python error:", err.toString()));

  py.on("close", async () => {
    try {
      const contextChunks = JSON.parse(data);
      const context = contextChunks.join("\n\n");

      const response = await axios.post("http://localhost:11434/api/generate", {
        model: "deepseek-r1:1.5b",
        prompt: `${context}\n\nQuestion: ${question}\nAnswer:`,
        stream: false
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
