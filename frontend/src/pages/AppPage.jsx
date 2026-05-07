import { useRef, useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Activity, Camera as CameraIcon, Settings, Settings2, Volume2, VolumeX, Hand } from 'lucide-react';
import '../index.css';

const LANGUAGE_OPTIONS = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'Hindi' },
  { code: 'es', label: 'Spanish' },
];

const QUICK_MESSAGES = [
  'I need help.',
  'Please wait.',
  'Thank you.',
  'Yes',
];

const MOBILE_CAM_KEYWORDS = ['droidcam', 'iriun', 'epoccam', 'phone', 'android', 'iphone', 'obs'];

export default function AppPage() {
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [history, setHistory] = useState([]);
  const [gesture, setGesture] = useState("Neutral");
  const [mar, setMar] = useState(0);
  const [spread, setSpread] = useState(0);
  const [confidence, setConfidence] = useState(0);
  const [language, setLanguage] = useState('en');
  const [highContrast, setHighContrast] = useState(false);
  const [fontScale, setFontScale] = useState(1);
  const [cameraDevices, setCameraDevices] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [cameraStatus, setCameraStatus] = useState('No camera selected');

  // Gestures & Voice
  const [handGesture, setHandGesture] = useState("None");
  const [facialGesture, setFacialGesture] = useState("Neutral");
  const [isMuted, setIsMuted] = useState(false);
  const isMutedRef = useRef(false);
  const [isTranscriptPaused, setIsTranscriptPaused] = useState(false);
  const lastSpokenHandRef = useRef("");

  const toggleMute = () => {
    const newState = !isMuted;
    setIsMuted(newState);
    isMutedRef.current = newState;
    if (newState) {
      window.speechSynthesis.cancel();
    }
  };

  const drawLandmarks = useCallback((landmarks) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');

    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#10b981'; // Success green
    landmarks.forEach(pt => {
      // CSS scaleX(-1) mirrors the canvas, so we use actual coordinates here!
      const x = pt.x * canvas.width;
      const y = pt.y * canvas.height;

      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  }, []);

  useEffect(() => {
    const connectWs = () => {
      let wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/stream';

      // Auto-upgrade to wss if frontend is on https and wsUrl is insecure
      if (window.location.protocol === 'https:' && wsUrl.startsWith('ws://')) {
        // Only upgrade if it's not localhost (to allow local dev on https)
        if (!wsUrl.includes('localhost') && !wsUrl.includes('127.0.0.1')) {
          wsUrl = wsUrl.replace('ws://', 'wss://');
        }
      }

      // Replace http/https with ws/wss if mistakenly provided
      if (wsUrl.startsWith('http://')) wsUrl = wsUrl.replace('http://', 'ws://');
      if (wsUrl.startsWith('https://')) wsUrl = wsUrl.replace('https://', 'wss://');

      console.log('Connecting to WebSocket:', wsUrl);
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket Connected');
        setIsConnected(true);
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.predicted_text) {
          setTranscript(data.predicted_text);
          setHistory((prev) => [data.predicted_text, ...prev].slice(0, 7));
        } else if (data.partial_text) {
          setTranscript(data.partial_text);
        }
        if (data.new_word && !isMutedRef.current) {
          const u = new SpeechSynthesisUtterance(data.new_word);
          window.speechSynthesis.speak(u);
        }

        setHandGesture(data.hand_gesture || "None");
        setFacialGesture(data.facial_gesture || "Neutral");
        setMar(data.mar || 0);
        setSpread(data.spread || 0);
        setConfidence(data.confidence || 0);

        if (data.landmarks && canvasRef.current && videoRef.current) {
          drawLandmarks(data.landmarks);
        }
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket Disconnected');
        setIsConnected(false);
        setTimeout(connectWs, 3000);
      };
    };

    connectWs();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [drawLandmarks]);

  const refreshCameraDevices = useCallback(async () => {
    try {
      const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      tempStream.getTracks().forEach((track) => track.stop());

      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === 'videoinput');
      setCameraDevices(videoInputs);

      if (videoInputs.length === 0) {
        setCameraStatus('No camera devices found');
        setSelectedCameraId('');
        return;
      }

      const preferred = videoInputs.find((d) =>
        MOBILE_CAM_KEYWORDS.some((keyword) => (d.label || '').toLowerCase().includes(keyword))
      );
      const fallback = videoInputs[0];
      const next = preferred || fallback;

      setSelectedCameraId((prev) => (prev && videoInputs.some((d) => d.deviceId === prev) ? prev : next.deviceId));
      setCameraStatus(`Found ${videoInputs.length} camera device(s)`);
    } catch (err) {
      console.error('Error listing camera devices:', err);
      setCameraStatus('Camera permission needed');
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      refreshCameraDevices();
    }, 0);
    return () => clearTimeout(timer);
  }, [refreshCameraDevices]);

  // Handle Hand Gesture Voice Synthesis
  useEffect(() => {
    if (handGesture !== 'None' && handGesture !== lastSpokenHandRef.current) {
      if (!isMutedRef.current) {
        let textToSpeak = "";
        if (handGesture === 'Thumb_Up') textToSpeak = "Yes";
        else if (handGesture === 'Thumb_Down') textToSpeak = "No";
        else if (handGesture === 'Open_Palm') textToSpeak = "Stop";
        else if (handGesture === 'Victory') textToSpeak = "Thank you";

        if (textToSpeak) {
          const u = new SpeechSynthesisUtterance(textToSpeak);
          window.speechSynthesis.speak(u);
        }
      }
      lastSpokenHandRef.current = handGesture;
    } else if (handGesture === 'None') {
      lastSpokenHandRef.current = "";
    }
  }, [handGesture, isMuted]);

  useEffect(() => {
    let intervalId;

    if (isStreaming && isConnected && !isTranscriptPaused) {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      intervalId = setInterval(() => {
        if (videoRef.current && videoRef.current.readyState === 4) {
          canvas.width = videoRef.current.videoWidth;
          canvas.height = videoRef.current.videoHeight;
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

          const base64Data = canvas.toDataURL('image/jpeg', 0.5);

          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
              frame: base64Data,
              language,
            }));
          }
        }
      }, 100);
    }

    return () => clearInterval(intervalId);
  }, [isStreaming, isConnected, language, isTranscriptPaused]);

  const startCamera = async () => {
    try {
      if (!selectedCameraId) {
        await refreshCameraDevices();
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: selectedCameraId ? { deviceId: { exact: selectedCameraId } } : true,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
        const activeLabel = stream.getVideoTracks()[0]?.label || 'Unknown camera';
        setCameraStatus(`Using: ${activeLabel}`);
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
      setCameraStatus('Camera stopped');

      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  };

  return (
    <div className={`app-container ${highContrast ? 'high-contrast' : ''}`} style={{ fontSize: `${fontScale}rem` }}>
      <header className="app-header">
        <button className="back-btn" onClick={() => navigate('/')}>
          <ArrowLeft size={20} /> Back
        </button>
        <div className="logo">
          <Activity className="logo-icon" size={24} />
          SilentSync AI
        </div>
        <div className="status-badge">
          <div className={`status-dot ${isConnected ? 'connected' : ''}`}></div>
          {isConnected ? 'Connected' : 'Connecting'}
        </div>
      </header>

      <motion.main
        className="main-content"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        <motion.section variants={itemVariants} className="camera-section glass-panel">
          <div className="section-header">
            <CameraIcon size={20} className="section-icon" />
            <h2>Camera Feed</h2>
          </div>

          <div className="camera-picker-row">
            <div style={{ flex: 1 }}>
              <select
                id="cameraSelect"
                className="ui-select"
                value={selectedCameraId}
                onChange={(e) => setSelectedCameraId(e.target.value)}
                disabled={isStreaming}
              >
                {cameraDevices.map((device, index) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label || `Camera ${index + 1}`}
                  </option>
                ))}
              </select>
            </div>
            <button className="btn compact" onClick={refreshCameraDevices} disabled={isStreaming}>
              Refresh
            </button>
          </div>
          <div className="camera-status">{cameraStatus}</div>
          <div className="video-container">
            <div className="video-wrapper">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
              />
              <canvas ref={canvasRef} style={{ pointerEvents: 'none', zIndex: 10 }} />
            </div>

            <div className="overlay-controls">
              {!isStreaming ? (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="btn primary"
                  onClick={startCamera}
                >
                  Start Tracking
                </motion.button>
              ) : (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="btn danger"
                  onClick={stopCamera}
                >
                  Stop Camera
                </motion.button>
              )}
            </div>
          </div>
        </motion.section>

        <section className="sidebar">
          <motion.div variants={itemVariants} className="glass-panel">
            <div className="section-header" style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                <Activity size={20} className="section-icon" />
                <h2 className="card-title">Live Transcript</h2>
              </div>
              <button
                onClick={() => setIsTranscriptPaused(!isTranscriptPaused)}
                className={`btn compact ${isTranscriptPaused ? 'primary' : 'outline'}`}
                style={{ padding: '0.3rem 0.6rem', fontSize: '0.8rem' }}
              >
                {isTranscriptPaused ? "Resume" : "Pause Tracking"}
              </button>
            </div>
            <div className={`transcript-box ${transcript ? 'active pulse' : ''}`} style={{ opacity: isTranscriptPaused ? 0.5 : 1 }}>
              {isTranscriptPaused ? "Tracking Paused..." : (transcript || "Waiting for speech...")}
            </div>
            <div className="history-list">
              {history.map((item, idx) => (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="history-item"
                  key={`${item}-${idx}`}
                >
                  {item}
                </motion.div>
              ))}
            </div>
          </motion.div>

          <motion.div variants={itemVariants} className="glass-panel">
            <div className="section-header" style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                <Hand size={20} className="section-icon" />
                <h2 className="card-title">AI Gestures & Voice</h2>
              </div>
              <button
                onClick={toggleMute}
                className="btn outline compact"
                style={{ border: 'none', padding: '5px' }}
                title={isMuted ? "Unmute Voice Assistant" : "Mute Voice Assistant"}
              >
                {isMuted ? <VolumeX size={20} color="#ef4444" /> : <Volume2 size={20} color="#10b981" />}
              </button>
            </div>

            <div className="gesture-status">
              Hand: <strong>{handGesture.replace("_", " ")}</strong> <br />
              Face: <strong>{facialGesture}</strong>
            </div>
          </motion.div>

        </section>
      </motion.main>
    </div>
  );
}
