import streamlit as st
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from audiorecorder import audiorecorder
from PIL import Image
from gtts import gTTS
import io
import os 

# --- 1. إعدادات الهوية البصرية ---
try:
    img = Image.open("logo.png")
    st.set_page_config(page_title="Speechify AI", page_icon=img, layout="wide")
except Exception:
    st.set_page_config(page_title="Speechify AI", layout="wide")

# --- 2. دوال المساعدة ---
def speak_text(text):
    """تحويل النص إلى صوت"""
    tts = gTTS(text=text, lang='ar')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def get_features(audio_data, sr):
    """استخراج الخصائص الصوتية"""
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# تعريف حالة الجلسة قبل أي استخدام
if 'total_xp' not in st.session_state:
    st.session_state.total_xp = 0

# --- 3. واجهة المستخدم (Sidebar) ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png")
    
    # اسم مالك التطبيق
    st.markdown("<h3 style='text-align: center; color: #4A90E2;'>رانيهان لطفي</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9em;'>مؤسس ومالك تطبيق Speechify AI</p>", unsafe_allow_html=True)
    st.divider()
    
    st.title("🚀 لوحة التحكم")
    st.metric("نقاط الخبرة (XP)", st.session_state.total_xp)

# الواجهة الرئيسية
st.title("مرحباً بك في Speechify AI 🗣️")
st.info("نحن هنا لنساعدك على إتقان مخارج الحروف العربية بكل سهولة ومرح. ابدأ تمرينك الآن!")

tab1, tab2, tab3 = st.tabs(["🎯 تمرين النطق", "📖 الدليل", "🛡️ الخصوصية"])

with tab1:
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        target_letter = st.selectbox("اختر الحرف المستهدف:", ["راء", "سين", "صاد"])
        st.write(f"لنتدرب على حرف **({target_letter})**")
        
        if st.button(f"🔊 اسمع نطق حرف ({target_letter})"):
            audio_fp = speak_text(target_letter)
            st.audio(audio_fp, format='audio/mp3')
            
    with col_r:
        # إصلاح السطر الذي سبب الخطأ (تأكد أن النص في سطر واحد)
        if target_letter == "راء":
            st.warning("نصيحة: تأكد من ملامسة طرف اللسان لسقف الحنك العلوي.")
        elif target_letter == "سين":
            st.warning("نصيحة: ضع طرف اللسان خلف الأسنان السفلى.")

    st.divider()
    st.subheader("🎤 سجل نطقك للحرف:")
    user_audio = audiorecorder("اضغط
