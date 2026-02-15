import { useEffect, useMemo, useState } from "react";

const DEFAULT_PROMPT =
  "Analyze this full song and provide concise, timestamped sections describing vocals, instrumentation, production effects, mix changes, energy flow, and genre cues. End with a short overall summary.";

export default function App() {
  const [mode, setMode] = useState("path");
  const [audioPath, setAudioPath] = useState("E:\\Coding\\hf-music-gen\\train-dataset\\Andrew Spacey - Wonder (Prod Beat It AT).mp3");
  const [audioFile, setAudioFile] = useState(null);
  const [backend, setBackend] = useState("hf_endpoint");
  const [endpointUrl, setEndpointUrl] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [modelId, setModelId] = useState("nvidia/audio-flamingo-3-hf");
  const [openAiApiKey, setOpenAiApiKey] = useState("");
  const [openAiModel, setOpenAiModel] = useState("gpt-5-mini");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [userContext, setUserContext] = useState("");
  const [artistName, setArtistName] = useState("");
  const [trackName, setTrackName] = useState("");
  const [enableWebSearch, setEnableWebSearch] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    let mounted = true;
    fetch("/api/config")
      .then((r) => r.json())
      .then((data) => {
        if (!mounted) return;
        const d = data?.defaults || {};
        if (d.backend) setBackend(d.backend);
        if (d.endpoint_url) setEndpointUrl(d.endpoint_url);
        if (d.model_id) setModelId(d.model_id);
        if (d.openai_model) setOpenAiModel(d.openai_model);
        if (d.af3_prompt) setPrompt(d.af3_prompt);
      })
      .catch(() => {});
    return () => {
      mounted = false;
    };
  }, []);

  const requestPreview = useMemo(() => {
    return {
      backend,
      endpoint_url: endpointUrl || "(env default)",
      model_id: modelId,
      openai_model: openAiModel,
      enable_web_search: enableWebSearch,
      artist_name: artistName || "(none)",
      track_name: trackName || "(none)",
    };
  }, [backend, endpointUrl, modelId, openAiModel, enableWebSearch, artistName, trackName]);

  async function runPipeline() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      let response;
      if (mode === "path") {
        response = await fetch("/api/pipeline/run-path", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            audio_path: audioPath,
            backend,
            endpoint_url: endpointUrl,
            hf_token: hfToken,
            model_id: modelId,
            af3_prompt: prompt,
            openai_api_key: openAiApiKey,
            openai_model: openAiModel,
            user_context: userContext,
            artist_name: artistName,
            track_name: trackName,
            enable_web_search: enableWebSearch,
          }),
        });
      } else {
        if (!audioFile) {
          throw new Error("Select an audio file first.");
        }
        const form = new FormData();
        form.append("audio_file", audioFile);
        form.append("backend", backend);
        form.append("endpoint_url", endpointUrl);
        form.append("hf_token", hfToken);
        form.append("model_id", modelId);
        form.append("af3_prompt", prompt);
        form.append("openai_api_key", openAiApiKey);
        form.append("openai_model", openAiModel);
        form.append("user_context", userContext);
        form.append("artist_name", artistName);
        form.append("track_name", trackName);
        form.append("enable_web_search", String(enableWebSearch));
        response = await fetch("/api/pipeline/run-upload", {
          method: "POST",
          body: form,
        });
      }

      const data = await response.json();
      if (!response.ok) {
        const detail = typeof data?.detail === "string" ? data.detail : JSON.stringify(data);
        throw new Error(detail);
      }
      setResult(data);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="hero">
        <h1>AF3 + ChatGPT Pipeline</h1>
        <p>Run Audio Flamingo 3 analysis, then clean/structure for Ace Step 1.5 LoRA metadata.</p>
      </div>

      <div className="grid">
        <section className="card">
          <h2>Inputs</h2>
          <div className="row">
            <label>Mode</label>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="path">Local Path</option>
              <option value="upload">Upload</option>
            </select>
          </div>

          {mode === "path" ? (
            <div className="row">
              <label>Audio Path</label>
              <input value={audioPath} onChange={(e) => setAudioPath(e.target.value)} />
            </div>
          ) : (
            <div className="row">
              <label>Audio File</label>
              <input type="file" accept="audio/*" onChange={(e) => setAudioFile(e.target.files?.[0] || null)} />
            </div>
          )}

          <div className="row">
            <label>AF3 Backend</label>
            <select value={backend} onChange={(e) => setBackend(e.target.value)}>
              <option value="hf_endpoint">HF Endpoint</option>
              <option value="local">Local Model</option>
            </select>
          </div>
          <div className="row">
            <label>AF3 Endpoint URL</label>
            <input value={endpointUrl} onChange={(e) => setEndpointUrl(e.target.value)} placeholder="https://..." />
          </div>
          <div className="row">
            <label>HF Token (optional)</label>
            <input type="password" value={hfToken} onChange={(e) => setHfToken(e.target.value)} />
          </div>
          <div className="row">
            <label>AF3 Model ID</label>
            <input value={modelId} onChange={(e) => setModelId(e.target.value)} />
          </div>
          <div className="row">
            <label>OpenAI API Key (optional)</label>
            <input type="password" value={openAiApiKey} onChange={(e) => setOpenAiApiKey(e.target.value)} />
          </div>
          <div className="row">
            <label>OpenAI Model</label>
            <input value={openAiModel} onChange={(e) => setOpenAiModel(e.target.value)} />
          </div>
          <div className="row">
            <label>Artist (optional)</label>
            <input value={artistName} onChange={(e) => setArtistName(e.target.value)} />
          </div>
          <div className="row">
            <label>Track (optional)</label>
            <input value={trackName} onChange={(e) => setTrackName(e.target.value)} />
          </div>
          <div className="row">
            <label>Prompt</label>
            <textarea rows={5} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </div>
          <div className="row">
            <label>User Context</label>
            <textarea rows={4} value={userContext} onChange={(e) => setUserContext(e.target.value)} />
          </div>
          <div className="row inline">
            <input
              id="websearch"
              type="checkbox"
              checked={enableWebSearch}
              onChange={(e) => setEnableWebSearch(e.target.checked)}
            />
            <label htmlFor="websearch">Enable ChatGPT web search (optional)</label>
          </div>

          <button className="run" disabled={loading} onClick={runPipeline}>
            {loading ? "Running..." : "Run Pipeline"}
          </button>
        </section>

        <section className="card">
          <h2>Request Summary</h2>
          <pre>{JSON.stringify(requestPreview, null, 2)}</pre>
          {error ? <p className="error">{error}</p> : null}
          {result ? (
            <>
              <h3>Saved Sidecar</h3>
              <p className="mono">{result.saved_to}</p>
              <h3>AF3 Analysis</h3>
              <pre>{result.af3_analysis}</pre>
              <h3>Final LoRA JSON</h3>
              <pre>{JSON.stringify(result.sidecar, null, 2)}</pre>
            </>
          ) : null}
        </section>
      </div>
    </div>
  );
}
