# voice_interface.py - Robust Voice Interface with Debugging

import streamlit as st
from voice_engine import AthenaVoice
import tempfile
import os
from datetime import datetime
import base64
import io
import wave


class VoiceInterface:
    """Streamlit-compatible voice interface with robust file handling"""
    
    def __init__(self):
        if 'voice_engine' not in st.session_state:
            with st.spinner("🎙️ Loading voice engine..."):
                try:
                    st.session_state.voice_engine = AthenaVoice(
                        whisper_model="base",
                        tts_lang='en'
                    )
                    st.session_state.voice_ready = True
                except Exception as e:
                    st.error(f"❌ Failed to load voice engine: {e}")
                    st.session_state.voice_ready = False
        
        self.voice = st.session_state.voice_engine if st.session_state.get('voice_ready') else None
    
    def save_uploaded_audio(self, audio_data) -> str:
        """Save audio data to a proper WAV file"""
        try:
            # Create persistent temp directory
            temp_dir = os.path.join(tempfile.gettempdir(), "athena_voice")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            audio_path = os.path.join(temp_dir, f"recording_{timestamp}.wav")
            
            # Write audio data
            audio_bytes = audio_data.getvalue() if hasattr(audio_data, 'getvalue') else audio_data
            
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
            
            # Verify file was created
            if not os.path.exists(audio_path):
                st.error(f"❌ Failed to create file at: {audio_path}")
                return None
            
            file_size = os.path.getsize(audio_path)
            
            if file_size == 0:
                st.error("❌ Audio file is empty (0 bytes)")
                return None
            
            st.success(f"✅ Audio saved: {file_size:,} bytes")
            print(f"✅ Audio file created:")
            print(f"   Path: {audio_path}")
            print(f"   Size: {file_size:,} bytes")
            print(f"   Exists: {os.path.exists(audio_path)}")
            
            return audio_path
            
        except Exception as e:
            st.error(f"❌ Error saving audio: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    def transcribe_audio_safe(self, audio_path: str) -> dict:
        """Safely transcribe audio with comprehensive error checking"""
        
        # Check 1: File exists
        if not os.path.exists(audio_path):
            return {
                'success': False,
                'text': '',
                'error': f'File not found: {audio_path}'
            }
        
        # Check 2: File size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            return {
                'success': False,
                'text': '',
                'error': 'Audio file is empty (0 bytes)'
            }
        
        st.info(f"📊 File ready: {os.path.basename(audio_path)} ({file_size:,} bytes)")
        
        # Check 3: Voice engine ready
        if not self.voice:
            return {
                'success': False,
                'text': '',
                'error': 'Voice engine not initialized'
            }
        
        # Transcribe
        try:
            with st.spinner("🎧 Transcribing (this may take 10-30 seconds)..."):
                result = self.voice.transcribe_audio(audio_path)
            
            if result.get('success'):
                st.success(f"✅ Transcribed: {result.get('confidence', 0):.0%} confidence")
                return result
            else:
                st.error(f"❌ Transcription failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            error_msg = f"Transcription exception: {str(e)}"
            st.error(f"❌ {error_msg}")
            import traceback
            st.code(traceback.format_exc())
            
            return {
                'success': False,
                'text': '',
                'error': error_msg
            }
    
    def speak_response(self, text: str) -> str:
        """Generate speech with error handling"""
        if not self.voice:
            st.error("❌ Voice engine not available")
            return None
        
        # Limit text length for faster TTS
        if len(text) > 500:
            text = text[:500] + "..."
            st.info("📝 Response truncated to 500 characters for voice")
        
        try:
            with st.spinner("🔊 Generating speech (requires internet)..."):
                audio_file = self.voice.speak(text)
            
            if audio_file and os.path.exists(audio_file):
                st.success("✅ Voice response generated!")
                return audio_file
            else:
                st.warning("⚠️ TTS failed. Check internet connection.")
                return None
                
        except Exception as e:
            st.error(f"❌ TTS error: {e}")
            return None
    
    def play_audio(self, audio_file: str):
        """Display audio player"""
        if audio_file and os.path.exists(audio_file):
            try:
                with open(audio_file, 'rb') as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format='audio/mp3')
            except Exception as e:
                st.error(f"❌ Error playing audio: {e}")


def render_voice_tab():
    """Voice interface with robust error handling"""
    
    st.markdown("### 🎙️ Voice Assistant")
    st.markdown("**Ask questions using your voice!**")
    
    # Check dependencies
    try:
        voice_interface = VoiceInterface()
        
        if not st.session_state.get('voice_ready'):
            st.error("❌ Voice engine failed to initialize")
            st.info("💡 Make sure you have installed: `pip install openai-whisper gtts`")
            return
            
    except Exception as e:
        st.error(f"❌ Voice interface error: {e}")
        st.info("Install dependencies: `pip install openai-whisper gtts`")
        return
    
    # Voice history
    if 'voice_history' not in st.session_state:
        st.session_state.voice_history = []
    
    # Settings
    with st.expander("⚙️ Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            auto_play = st.checkbox("🔊 Auto-play", value=True)
            query_mode = st.selectbox(
                "Mode",
                ["💬 Chat", "🔍 Search", "❓ Q&A"],
                help="Chat: Conversation | Search: Find sections | Q&A: Precise answers"
            )
        
        with col2:
            show_text = st.checkbox("📝 Show text", value=True)
    
    if st.button("🗑️ Clear History"):
        st.session_state.voice_history = []
        st.success("Cleared!")
        st.rerun()
    
    st.markdown("---")
    
    # History
    if st.session_state.voice_history:
        st.markdown("### 💬 Conversation History")
        
        for exchange in st.session_state.voice_history[-3:]:  # Show last 3
            with st.container():
                st.markdown(f"**🎤 You ({exchange['timestamp']}):** {exchange['question'][:100]}")
                
                if show_text:
                    with st.expander("View Response"):
                        st.info(exchange['response'][:300] + '...' if len(exchange['response']) > 300 else exchange['response'])
                
                if exchange.get('audio_file') and os.path.exists(exchange['audio_file']):
                    st.audio(exchange['audio_file'])
                
                st.markdown("---")
    
    # Input section
    st.markdown("### 🎤 Record Your Question")
    
    # Mode info
    mode_help = {
        "💬 Chat": "💬 Natural conversation about the document",
        "🔍 Search": "🔍 Find specific sections or topics",
        "❓ Q&A": "❓ Get precise factual answers"
    }
    st.info(mode_help[query_mode])
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Audio input
        st.markdown("**Record Audio:**")
        audio_data = st.audio_input("Click to record")
    
    with col2:
        st.markdown("**Or type:**")
        text_input = st.text_input("Question", label_visibility="collapsed", 
                                   placeholder="Type here...")
    
    # Process audio
    if audio_data is not None:
        st.success("✅ Audio recorded!")
        
        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write(f"Audio data type: {type(audio_data)}")
            st.write(f"Audio data size: {len(audio_data.getvalue())} bytes")
        
        if st.button("🎧 Transcribe & Answer", key="process_audio", type="primary"):
            # Save audio
            audio_path = voice_interface.save_uploaded_audio(audio_data)
            
            if not audio_path:
                st.error("❌ Failed to save audio file")
                return
            
            # Transcribe
            result = voice_interface.transcribe_audio_safe(audio_path)
            
            if result['success'] and result['text'].strip():
                question = result['text']
                st.markdown(f"**📝 You asked:** {question}")
                
                # Process query
                process_query(question, voice_interface, query_mode, auto_play)
                
                # Cleanup
                try:
                    os.unlink(audio_path)
                except:
                    pass
            else:
                st.error(f"❌ Transcription failed: {result.get('error', 'Unknown error')}")
                st.info("💡 Try speaking more clearly or check your microphone")
    
    # Process text input
    if text_input.strip():
        if st.button("💬 Answer (with voice)", key="process_text", type="primary"):
            process_query(text_input, voice_interface, query_mode, auto_play)


def process_query(question: str, voice_interface: VoiceInterface, 
                 query_mode: str, auto_play: bool):
    """Process query and generate response"""
    
    # Get response based on mode
    if query_mode == "💬 Chat":
        response_text = get_chat_response(question)
        mode_label = "Chat"
    elif query_mode == "🔍 Search":
        response_text = get_search_response(question)
        mode_label = "Search"
    else:
        response_text = get_qa_response(question)
        mode_label = "Q&A"
    
    if not response_text:
        return
    
    # Display response
    st.markdown(f"**🧠 Athena ({mode_label}):**")
    with st.expander("View Full Response", expanded=True):
        st.success(response_text)
    
    # Generate voice
    audio_file = voice_interface.speak_response(response_text)
    
    if audio_file:
        # Save to history
        st.session_state.voice_history.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'question': question,
            'response': response_text,
            'audio_file': audio_file,
            'mode': mode_label
        })
        
        # Play
        if auto_play:
            st.markdown("**🔊 Playing response...**")
            voice_interface.play_audio(audio_file)
        else:
            st.markdown("**🔊 Click to play:**")
            voice_interface.play_audio(audio_file)


def get_chat_response(question: str) -> str:
    """Get chat response"""
    if 'athena_chat' not in st.session_state:
        st.error("❌ Upload a document first!")
        return None
    
    with st.spinner("💭 Thinking..."):
        return st.session_state.athena_chat.chat(question)


def get_search_response(question: str) -> str:
    """Get semantic search response"""
    if 'semantic_index' not in st.session_state:
        st.error("❌ Build semantic index first (Semantic Search tab)")
        return None
    
    with st.spinner("🔍 Searching..."):
        from semantic_search import search_semantic
        results = search_semantic(st.session_state.semantic_index, question, k=3)
        
        if not results:
            return "No relevant sections found."
        
        response = f"Found {len(results)} relevant sections:\n\n"
        for i, (text, score) in enumerate(results, 1):
            response += f"{i}. ({score:.0%} match)\n{text[:150]}...\n\n"
        
        return response


def get_qa_response(question: str) -> str:
    """Get Q&A response"""
    if 'qa_chain' not in st.session_state:
        st.error("❌ Build Q&A index first (Q&A tab)")
        return None
    
    with st.spinner("❓ Finding answer..."):
        return st.session_state.qa_chain(question)


if __name__ == "__main__":
    st.set_page_config(page_title="Voice Test", page_icon="🎤")
    st.title("🎤 Voice Interface Test")
    render_voice_tab()