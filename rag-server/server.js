const express = require("express");
const { spawn } = require("child_process");

const app = express();
app.use(express.json());

app.post("/rag", async (req, res) => {
  const { question, top_k = 5 } = req.body;

  const embed = spawn("python", ["embed.py", question, top_k.toString()]);

  let embedData = "";
  embed.stdout.on("data", (chunk) => {
    embedData += chunk.toString();
  });

  embed.stderr.on("data", (err) => {
    console.error("Embed Python error:", err.toString());
  });
  console.log("Embed Python er");
  embed.on("close", () => {
    try {
      const contextChunks = JSON.parse(embedData);
      const context = contextChunks.join("\n\n");
      console.log("Embed Python e2r");
      const generate = spawn("python", ["generate.py", question, context]);

      let output = "";
      generate.stdout.on("data", (chunk) => {
        output += chunk.toString();
      });
      console.log("Embed Python e3r");
      generate.stderr.on("data", (err) => {
        console.error("Generate Python error:", err.toString());
      });
      console.log("Embed Python e4r");
      generate.on("close", () => {
      try {
        console.log(output);

        // Try to extract the first valid JSON object in the output
        const firstBrace = output.indexOf('{');
        const lastBrace = output.lastIndexOf('}');

        if (firstBrace !== -1 && lastBrace !== -1 && firstBrace < lastBrace) {
          const jsonString = output.slice(firstBrace, lastBrace + 1);

          try {
            // Sanitize the JSON string if needed
            const cleaned = jsonString.replace(/[\x00-\x1F\x7F]/g, "");  // remove control characters
            const parsed = JSON.parse(cleaned);

            const answerText = parsed["DeepSeek Answer"] || parsed.answer || parsed.response;
            res.json({ answer: answerText });
          } catch (e) {
            console.error("JSON parse error:", e);
            res.status(500).json({ error: "Failed to parse JSON block" });
          }
        } else {
          res.status(500).json({ error: "Could not locate valid JSON block" });
        }
      } catch (err) {
        console.error("Unexpected error:", err);
        res.status(500).json({ error: "Unexpected error occurred" });
      }

         
      });
    } catch (err) {
      console.error("Embed parse error:", err);
      res.status(500).json({ error: "Failed to parse context." });
    }
  });
});

app.listen(3000, () => {
  console.log("🔊 Local RAG backend running on http://localhost:3000");
});
