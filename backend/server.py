#!/usr/bin/env python3

"""
AI-Powered Aggression Detection Server

Handles real-time audio and facial analysis for emotion detection
"""

import asyncio
import json
import logging
import numpy as np
import time
import websockets
from collections import deque
import base64
import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from email_notifier import EmailNotifier

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for storing analysis data
facial_data_buffer = deque(maxlen=10)
audio_data_buffer = deque(maxlen=50)
analysis_results = {}
audio_feature_buffer = deque(maxlen=20)  # ~ last 20 secs if cadence is ~1 Hz

class EmotionAnalyzer:
    """AI-powered emotion analyzer for combined audio and facial analysis"""
    
    def __init__(self):
        self.facial_weights = {
            'angry': 0.7,
            'fearful': 0.2,
            'disgusted': 0.1
        }
        logger.info("Emotion analyzer initialized.")
        
    def analyze_facial_emotions(self, expressions):
        """Analyze facial expressions for aggression indicators"""
        if not expressions:
            return {'aggression_score': 0, 'confidence': 0, 'emotions': {}}
            
        aggression_score = 0
        for emotion, weight in self.facial_weights.items():
            if emotion in expressions:
                aggression_score += expressions[emotion] * weight * 100
                
        if aggression_score < 10 and (
            expressions.get('angry', 0) > 0.005 or 
            expressions.get('fearful', 0) > 0.005 or
            expressions.get('disgusted', 0) > 0.005
        ):
            aggression_score = 10
                
        max_expression = max(expressions.values()) if expressions else 0
        confidence = min(max_expression * 100, 95)
        emotion_bars = {k: round(v * 100, 2) for k, v in expressions.items()}
        
        return {
            'aggression_score': min(aggression_score, 100),
            'confidence': confidence,
            'emotions': expressions,
            'emotion_bars': emotion_bars
        }
        
    def analyze_audio_emotions(self, audio_data):
        """Set voice emotion output to match the current facial emotion output."""
        if not audio_data:
            return {'voice_emotion': 'Unknown', 'aggression_score': 0, 'confidence': 0, 'audio_bars': {}, 'multi_voice': False}
        
        try:
            # If facial result exists, use it for voice output
            facial_result = analysis_results.get('facial')
            if facial_result and 'emotion_bars' in facial_result:
                emotion_bars = facial_result['emotion_bars']
                voice_emotion = max(emotion_bars, key=emotion_bars.get)
                aggression_score = int(emotion_bars.get("angry", 0) * 0.7 + emotion_bars.get("fearful", 0) * 0.2 + emotion_bars.get("disgusted", 0) * 0.1)
                confidence = int(max(emotion_bars.values()))
                return {
                    'voice_emotion': voice_emotion,
                    'aggression_score': aggression_score,
                    'confidence': confidence, 
                    'audio_bars': emotion_bars,
                    'multi_voice': False
                }
            
            # Fallback: neutral
            emotions = ["angry", "happy", "sad", "fearful", "disgusted", "surprised", "neutral"]
            audio_bars = {e: (100 if e == "neutral" else 0) for e in emotions}
            return {
                'voice_emotion': "neutral",
                'aggression_score': 0,
                'confidence': 100,
                'audio_bars': audio_bars,
                'multi_voice': False
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {'voice_emotion': 'Unknown', 'aggression_score': 0, 'confidence': 0, 'audio_bars': {}, 'multi_voice': False}

    def analyze_audio_features(self, features: dict):
        """Heuristic voice emotion estimation from lightweight audio features."""
        try:
            if not isinstance(features, dict):
                return {'voice_emotion': 'Unknown', 'aggression_score': 0, 'confidence': 0, 'audio_bars': {}, 'multi_voice': False}
            
            rms = float(features.get('rms', 0.0))
            zcr = float(features.get('zcr', 0.0))
            centroid = float(features.get('centroid', 0.0))
            rolloff = float(features.get('rolloff', 0.0))
            voiced = bool(features.get('voiced', False))
            pitch = float(features.get('pitch', 0.0))
            
            angry = max(0.0, 0.6*rms + 0.2*zcr + 0.2*centroid)
            happy = max(0.0, 0.4*rms + 0.4*centroid + 0.2*rolloff + 0.1*pitch)
            sad = max(0.0, 0.6*(1.0-rms) + 0.3*(1.0*centroid) + 0.1*(1.0*rolloff))
            fearful = max(0.0, 0.3*zcr + 0.4*centroid + 0.3*(1.0-rms))
            disgusted = max(0.0, 0.3*rms + 0.3*(1.0*centroid) + 0.4*zcr)
            surprised = max(0.0, 0.5*rolloff + 0.3*centroid + 0.2*rms)
            neutral = max(0.0, (0.8*(1.0-rms)) * (0.6 if voiced else 1.0))
            
            raw = {
                'angry': angry, 'happy': happy, 'sad': sad, 'fearful': fearful,
                'disgusted': disgusted, 'surprised': surprised, 'neutral': neutral
            }
            m = max(1e-6, max(raw.values()))
            bars = {k: round(100.0*v/m, 2) for k, v in raw.items()}
            voice_emotion = max(bars, key=bars.get)
            aggression_score = int(bars.get('angry', 0) * 0.7 + bars.get('fearful', 0) * 0.2 + bars.get('disgusted', 0) * 0.1)
            confidence = int(max(bars.values()))
            
            return {
                'voice_emotion': voice_emotion,
                'aggression_score': aggression_score,
                'confidence': confidence,
                'audio_bars': bars,
                'multi_voice': False
            }
        except Exception as e:
            logger.error(f"Error analyzing audio features: {e}")
            return {'voice_emotion': 'Unknown', 'aggression_score': 0, 'confidence': 0, 'audio_bars': {}, 'multi_voice': False}
    
    def combine_analysis(self, facial_result, audio_result):
        """Combine facial and audio analysis for comprehensive aggression detection"""
        facial_weight = 0.6
        audio_weight = 0.4
        
        facial_score = facial_result.get('aggression_score', 0)
        audio_score = audio_result.get('aggression_score', 0)
        combined_score = (facial_score * facial_weight) + (audio_score * audio_weight)
        
        facial_confidence = facial_result.get('confidence', 0)
        audio_confidence = audio_result.get('confidence', 0)
        combined_confidence = (facial_confidence * facial_weight) + (audio_confidence * audio_weight)
        
        if combined_score > 80:
            recommendation = "High risk - Immediate intervention recommended"
        elif combined_score > 60:
            recommendation = "Moderate risk - Monitor closely"
        elif combined_score > 40:
            recommendation = "Low risk - Continue monitoring"
        else:
            recommendation = "No action needed"
            
        return {
            'combined_score': round(combined_score, 1),
            'combined_confidence': round(combined_confidence, 1),
            'facial_score': facial_score,
            'audio_score': audio_score,
            'voice_emotion': audio_result.get('voice_emotion', 'Unknown'),
            'recommendation': recommendation
        }

# Initialize the emotion analyzer and email notifier
emotion_analyzer = EmotionAnalyzer()
email_notifier = EmailNotifier()

# Configure email settings
ALERT_EMAIL = "ayush.fireball2005@gmail.com"  # Recipient email address (where alerts will be sent)
ANGER_NOTIFICATION_COOLDOWN = 60  # Minimum seconds between email notifications
last_email_time = 0

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected from %s', request.remote_addr)
    emit('status', {
        'message': 'Connected to AI server',
        'status': 'success',
        'timestamp': time.time()
    })
    logger.info('Sent connection confirmation to client')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('anger_detected')
def handle_anger_detected(data):
    """Handle anger detection events and send email notifications"""
    global last_email_time
    
    current_time = time.time()
    if current_time - last_email_time >= ANGER_NOTIFICATION_COOLDOWN:
        anger_level = data.get('angerLevel', 0)
        threshold = data.get('threshold', 50)
        
        # Send email notification
        if email_notifier.send_alert(
            recipient_email=ALERT_EMAIL,
            anger_level=round(anger_level, 1)
        ):
            last_email_time = current_time
            emit('notification_sent', {
                'success': True,
                'message': 'Alert email sent successfully'
            })
        else:
            emit('notification_sent', {
                'success': False,
                'message': 'Failed to send alert email'
            })

@socketio.on('facial_analysis')
def handle_facial_analysis(data):
    """Handle incoming facial analysis data"""
    try:
        expressions = data.get('expressions', {})
        face_index = data.get('faceIndex', 0)
        facial_result = emotion_analyzer.analyze_facial_emotions(expressions)
        analysis_results[f'facial_face_{face_index}'] = facial_result
        
        all_facial_results = [v for k, v in analysis_results.items() if k.startswith('facial_face_') and isinstance(v, dict)]
        if all_facial_results:
            avg_aggression_score = sum(r.get('aggression_score', 0) for r in all_facial_results) / len(all_facial_results)
            avg_confidence = sum(r.get('confidence', 0) for r in all_facial_results) / len(all_facial_results)
            emotion_keys = set(k for r in all_facial_results for k in r.get('emotion_bars', {}).keys())
            avg_emotion_bars = {k: round(sum(r.get('emotion_bars', {}).get(k, 0) for r in all_facial_results) / len(all_facial_results), 2) for k in emotion_keys}
            
            aggregated_facial_result = {
                'aggression_score': avg_aggression_score,
                'confidence': avg_confidence,
                'emotions': expressions,
                'emotion_bars': avg_emotion_bars,
                'face_count': len(all_facial_results)
            }
            analysis_results['facial'] = aggregated_facial_result
        else:
            aggregated_facial_result = facial_result
            
        if 'audio' in analysis_results and analysis_results.get('audio'):
            combined_result = emotion_analyzer.combine_analysis(
                aggregated_facial_result,
                analysis_results['audio']
            )
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': combined_result['voice_emotion'],
                'combinedScore': combined_result['combined_score'],
                'confidence': combined_result['combined_confidence'],
                'recommendation': combined_result['recommendation'],
                'facialScore': combined_result['facial_score'],
                'audioScore': combined_result['audio_score'],
                'faceCount': aggregated_facial_result.get('face_count', 1),
                'emotionBars': aggregated_facial_result.get('emotion_bars', {}),
                'audioBars': analysis_results['audio'].get('audio_bars', {}),
                'multiVoice': int(analysis_results['audio'].get('multi_voice', False))
            })
        else:
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': 'Analyzing...',
                'combinedScore': aggregated_facial_result.get('aggression_score', 0),
                'confidence': aggregated_facial_result.get('confidence', 0),
                'recommendation': f'Monitoring {aggregated_facial_result.get("face_count", 1)} face(s)...',
                'facialScore': aggregated_facial_result.get('aggression_score', 0),
                'audioScore': 0,
                'faceCount': aggregated_facial_result.get('face_count', 1),
                'emotionBars': aggregated_facial_result.get('emotion_bars', {})
            })
            
    except Exception as e:
        logger.error(f"Error processing facial analysis: {e}", exc_info=True)

@socketio.on('audio')
def handle_audio_data(data):
    """Handle incoming audio data"""
    try:
        audio_data = data.get('data')
        audio_result = emotion_analyzer.analyze_audio_emotions(audio_data)
        analysis_results['audio'] = audio_result
        
        if 'facial' in analysis_results and analysis_results.get('facial'):
            combined_result = emotion_analyzer.combine_analysis(
                analysis_results['facial'],
                audio_result
            )
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': combined_result['voice_emotion'],
                'combinedScore': combined_result['combined_score'],
                'confidence': combined_result['combined_confidence'],
                'recommendation': combined_result['recommendation'],
                'facialScore': combined_result['facial_score'],
                'audioScore': combined_result['audio_score'],
                'faceCount': analysis_results['facial'].get('face_count', 1),
                'emotionBars': analysis_results['facial'].get('emotion_bars', {}),
                'audioBars': audio_result.get('audio_bars', {}),
                'multiVoice': int(audio_result.get('multi_voice', False))
            })
        else:
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': audio_result.get('voice_emotion', 'Unknown'),
                'combinedScore': audio_result.get('aggression_score', 0),
                'confidence': audio_result.get('confidence', 0),
                'recommendation': 'Monitoring voice patterns...',
                'facialScore': 0,
                'audioScore': audio_result.get('aggression_score', 0),
                'faceCount': 0,
                'emotionBars': {},
                'audioBars': audio_result.get('audio_bars', {}),
                'multiVoice': int(audio_result.get('multi_voice', False))
            })
            
    except Exception as e:
        logger.error(f"Error processing audio data: {e}", exc_info=True)

@socketio.on('audio_features')
def handle_audio_features(data):
    """Handle incoming audio features from client for voice emotion estimation."""
    try:
        features = data.get('features', {})
        frame_result = emotion_analyzer.analyze_audio_features(features)
        audio_feature_buffer.append({
            'ts': time.time(),
            'result': frame_result,
            'voiced': bool(features.get('voiced', False))
        })
        
        now = time.time()
        recent = [x for x in audio_feature_buffer if now - x['ts'] <= 3.0]
        if not recent:
            return
            
        keys = set()
        for r in recent:
            keys.update(r['result'].get('audio_bars', {}).keys())
        avg_bars = {}
        for k in keys:
            avg_bars[k] = round(sum(r['result'].get('audio_bars', {}).get(k, 0) for r in recent) / len(recent), 2)
            
        voiced_ratio = sum(1 for r in recent if r['voiced']) / len(recent)
        if voiced_ratio < 0.3:
            for k in avg_bars.keys():
                avg_bars[k] = 0.0
            avg_bars['neutral'] = 100.0
            
        stable_voice = max(avg_bars, key=avg_bars.get) if avg_bars else 'neutral'
        stable_conf = int(max(avg_bars.values())) if avg_bars else 50
        stable_aggr = int(avg_bars.get('angry', 0) * 0.7 + avg_bars.get('fearful', 0) * 0.2 + avg_bars.get('disgusted', 0) * 0.1)
        audio_result = {
            'voice_emotion': stable_voice,
            'aggression_score': stable_aggr,
            'confidence': stable_conf,
            'audio_bars': avg_bars,
            'multi_voice': False
        }
        
        analysis_results['audio'] = audio_result
        
        if 'facial' in analysis_results and analysis_results.get('facial'):
            combined_result = emotion_analyzer.combine_analysis(
                analysis_results['facial'],
                audio_result
            )
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': combined_result['voice_emotion'],
                'combinedScore': combined_result['combined_score'],
                'confidence': combined_result['combined_confidence'],
                'recommendation': combined_result['recommendation'],
                'facialScore': combined_result['facial_score'],
                'audioScore': combined_result['audio_score'],
                'faceCount': analysis_results['facial'].get('face_count', 1),
                'emotionBars': analysis_results['facial'].get('emotion_bars', {}),
                'audioBars': audio_result.get('audio_bars', {}),
                'multiVoice': int(audio_result.get('multi_voice', False))
            })
        else:
            emit('emotion_analysis', {
                'type': 'emotion_analysis',
                'voiceEmotion': audio_result.get('voice_emotion', 'Unknown'),
                'combinedScore': audio_result.get('aggression_score', 0),
                'confidence': audio_result.get('confidence', 0),
                'recommendation': 'Monitoring voice patterns...',
                'facialScore': 0,
                'audioScore': audio_result.get('aggression_score', 0),
                'faceCount': 0,
                'emotionBars': {},
                'audioBars': audio_result.get('audio_bars', {}),
                'multiVoice': int(audio_result.get('multi_voice', False))
            })
            
    except Exception as e:
        logger.error(f"Error processing audio features: {e}", exc_info=True)

@app.route('/')
def index():
    return "AI Server is running."

def run_server():
    """Run the Flask server with SocketIO"""
    try:
        logger.info("Starting AI Aggression Detection Server...")
        port = int(os.environ.get("PORT", 10000))
        logger.info(f"Server will listen on port {port}")
        logger.info("Emotion analyzer status: %s", "initialized" if emotion_analyzer else "not initialized")
        socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == '__main__':
    run_server()