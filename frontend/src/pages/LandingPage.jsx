import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Camera, Brain, MessageSquare, ChevronRight } from 'lucide-react';
import NetworkBackground from '../components/NetworkBackground';
import '../index.css';

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="landing-container">
      {/* Dynamic Neural Network Canvas */}
      <NetworkBackground />
      
      {/* Background Orbs */}
      <div className="orb orb-1"></div>
      <div className="orb orb-2"></div>
      <div className="orb orb-3"></div>

      <nav className="landing-nav">
        <div className="logo">
          <Brain className="logo-icon" />
          SilentSync AI
        </div>
      </nav>

      <main className="landing-content">
        <motion.div 
          className="hero-section"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <motion.div 
            className="badge"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
          >
            v2.0 Deep Learning Upgrade
          </motion.div>
          
          <h1 className="hero-title">
            Speak silently. <br/>
            <span className="gradient-text">We'll do the typing.</span>
          </h1>
          
          <p className="hero-subtitle">
            An advanced computer vision and PyTorch LSTM AI system that translates your lip movements into text in real-time. No microphone required.
          </p>

          <motion.button 
            className="cta-button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate('/app')}
          >
            Launch Web App <ChevronRight className="btn-icon" />
          </motion.button>
        </motion.div>

        <div className="features-grid">
          <FeatureCard 
            delay={0.4}
            icon={<Camera size={32} />}
            title="40-Point Tracking"
            desc="Extracts ultra-precise 3D facial landmarks from your camera in real-time."
          />
          <FeatureCard 
            delay={0.6}
            icon={<Brain size={32} />}
            title="LSTM Neural Network"
            desc="A custom PyTorch deep learning model that understands temporal sequences and sentences."
          />
          <FeatureCard 
            delay={0.8}
            icon={<MessageSquare size={32} />}
            title="Live Transcript"
            desc="Instantly streams predictions via WebSockets directly to your screen."
          />
        </div>
      </main>
    </div>
  );
}

function FeatureCard({ icon, title, desc, delay }) {
  return (
    <motion.div 
      className="feature-card glass-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ y: -10, boxShadow: '0 20px 40px rgba(16, 185, 129, 0.2)' }}
    >
      <div className="feature-icon">{icon}</div>
      <h3 className="feature-title">{title}</h3>
      <p className="feature-desc">{desc}</p>
    </motion.div>
  );
}
