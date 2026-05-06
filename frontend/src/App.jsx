import React, { useRef, useEffect, useState, useCallback } from 'react';
import './index.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [gesture, setGesture] = useState("Neutral");
  const [mar, setMar] = useState(0);

  // Initialize WebSocket
  useEffect(() => {
    const connectWs = () => {
      wsRef.current = new WebSocket('ws://localhost:8000/ws/stream');
      
      wsRef.current.onopen = () => {
        console.log('WebSocket Connected');
        setIsConnected(true);
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.predicted_text) {
          setTranscript(data.predicted_text);
        }
        setGesture(data.gesture || "Neutral");
        setMar(data.mar || 0);
        
        // Render landmarks
        if (data.landmarks && canvasRef.current && videoRef.current) {
          drawLandmarks(data.landmarks);
        }
      };
      
      wsRef.current.onclose = () => {
        console.log('WebSocket Disconnected');
        setIsConnected(false);
        setTimeout(connectWs, 3000); // Reconnect
      };
    };

    connectWs();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const drawLandmarks = useCallback((landmarks) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    
    // Match canvas size to video dimensions
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw landmarks
    ctx.fillStyle = '#10b981'; // Success green
    landmarks.forEach(pt => {
      // Note: Video is mirrored in CSS, so we need to mirror X coordinates on canvas too
      const x = canvas.width - (pt.x * canvas.width);
      const y = pt.y * canvas.height;
      
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  }, []);

  // Capture frames and send via WS
  useEffect(() => {
    let intervalId;
    
    if (isStreaming && isConnected) {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      intervalId = setInterval(() => {
        if (videoRef.current && videoRef.current.readyState === 4) {
          canvas.width = videoRef.current.videoWidth;
          canvas.height = videoRef.current.videoHeight;
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          
          // Get base64 string
          // Lower quality for performance over websocket
          const base64Data = canvas.toDataURL('image/jpeg', 0.5);
          
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(base64Data);
          }
        }
      }, 100); // ~10fps to backend
    }
    
    return () => clearInterval(intervalId);
  }, [isStreaming, isConnected]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Could not access camera. Please allow permissions.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
      
      // Clear canvas
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  };

  return (
    <div className="app-container">
      <header>
        <div className="logo">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="12" x2="12" y1="19" y2="22" />
          </svg>
          SilentSync AI
        </div>
        <div className="status-badge">
          <div className={`status-dot ${isConnected ? 'connected' : ''}`}></div>
          {isConnected ? 'Backend Connected' : 'Connecting...'}
        </div>
      </header>

      <main className="main-content">
        <section className="camera-section glass-panel">
          <div className="video-container">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted
            />
            <canvas ref={canvasRef} />
            
            <div className="overlay-controls">
              {!isStreaming ? (
                <button className="btn primary" onClick={startCamera}>
                  Start Tracking
                </button>
              ) : (
                <button className="btn" onClick={stopCamera}>
                  Stop Camera
                </button>
              )}
            </div>
          </div>
        </section>

        <section className="sidebar">
          <div className="glass-panel">
            <h2 className="card-title">Live Transcript</h2>
            <div className={`transcript-box ${transcript ? 'active pulse' : ''}`}>
              {transcript || "Waiting for speech..."}
            </div>
          </div>

          <div className="glass-panel">
            <h2 className="card-title">AI Analysis</h2>
            
            <div className="gesture-status" style={{ marginBottom: '1.5rem' }}>
              Current State: <strong style={{color: 'white'}}>{gesture}</strong>
            </div>

            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Mouth Aspect Ratio</div>
                <div className="metric-value">{mar.toFixed(2)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Status</div>
                <div className="metric-value" style={{fontSize: '1.2rem', marginTop: '1rem'}}>
                  {isStreaming ? 'Active' : 'Idle'}
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
