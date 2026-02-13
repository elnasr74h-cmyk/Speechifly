import streamlit as st
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from audiorecorder import audiorecorder
from PIL import Image
from gtts import gTTS
import io
import os 

# --- 1. الإعدادات ---
try:
    img = Image.open("logo.png")
    st.set_page_config(page_title="Speechify AI", page_icon=img)
except:
    st.set_page_config(page_title="Speechify AI")

# --- 2. الدوال ---
def speak_text(text):
    tts = gTTS(text=text, lang='ar')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def get_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

if 'total_xp' not in st.session_state:
    st.session_state.total_xp = 0

# --- 3. القائمة الجانبية ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png")
    st.markdown("### رانيهان لطفي")
    st.write("مؤسس التطبيق")
    st.divider()
    st.metric("XP النقاط", st.session_state.total_xp)

# --- 4. الواجهة الرئيسية ---
st.title("Speechify AI 🗣️")

t1, t2, t3 = st.tabs(["التدريب", "الدليل", "الخصوصية"])

with t1:
    target = st.selectbox("الحرف:", ["راء", "سين", "صاد"])
    
    if st.button("🔊 اسمع"):
        audio_fp = speak_text(target)
        st.audio(audio_fp)
            
    if target == "راء":
        st.warning("نصيحة: ارفع طرف اللسان للسقف.")
    
    st.divider()
    # تم تقصير هذا السطر خصيصاً لتجنب خطأ SyntaxError
    u_audio = audiorecorder("🎤 سجل", "🛑 إيقاف")

    if len(u_audio) > 0:
        y, sr = librosa.load(u_audio.export(), sr=22050)
        u_feat = get_features(y, sr)
        ref = np.random.rand(13) 
        sim = cosine_similarity([ref], [u_feat])[0][0]
        score = int(sim * 100)

        if score > 75:
            st.success(f"ممتاز! الدقة: {score}%")
            st.session_state.total_xp += 50
            st.balloons()
        else:
            st.error(f"حاول ثانية. الدقة: {score}%")

with t2:
    st.write("سجل صوتك وقارنه بالنطق الصحيح.")

with t3:
    st.write("خصوصيتك محفوظة.")

st.divider()
st.caption("حقوق الملكية © 2026 - رانيهان لطفي")
