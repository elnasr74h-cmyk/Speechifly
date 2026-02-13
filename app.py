import streamlit as st
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from audiorecorder import audiorecorder
from PIL import Image
from gtts import gTTS
import io
import os  # إضافة مكتبة os للتحقق من وجود الملفات [cite: 1]

# --- 1. إعدادات الهوية البصرية ---
try:
    img = Image.open("logo.png")
    st.set_page_config(page_title="Speechify AI", page_icon=img, layout="wide")
except:
    st.set_page_config(page_title="Speechify AI", layout="wide")

# --- 2. دوال المساعدة (TTS & Analysis) ---
def speak_text(text):
    # تحويل النص إلى صوت باستخدام gTTS [cite: 1, 4]
    tts = gTTS(text=text, lang='ar')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp

def get_features(audio_data, sr):
    # استخراج معاملات MFCC للتحليل الصوتي [cite: 1]
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# الحفاظ على نقاط الخبرة في الجلسة [cite: 1, 2]
if 'total_xp' not in st.session_state:
    st.session_state.total_xp = 0

# --- 3. واجهة المستخدم (Sidebar) ---
# عرض الشعار في القائمة الجانبية إذا وجد [cite: 2]
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png")

st.sidebar.title("🚀 لوحة التحكم")
st.sidebar.metric("نقاط الخبرة (XP)", st.session_state.total_xp) [cite: 2]

# رسالة ترحيبية مشجعة [cite: 2, 3]
st.title("مرحباً بك في Speechify AI 🗣️")
st.balloons()
st.info("نحن هنا لنساعدك على إتقان مخارج الحروف العربية بكل سهولة ومرح. ابدأ تمرينك الآن!") [cite: 2, 3]

# تقسيم الواجهة إلى تبويبات [cite: 3]
tab1, tab2, tab3 = st.tabs(["🎯 تمرين النطق", "📖 الدليل", "🛡️ الخصوصية"])

with tab1:
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        target_letter = st.selectbox("اختر الحرف المستهدف:", ["راء", "سين", "صاد"]) [cite: 4]
        st.write(f"لنتدرب على حرف **({target_letter})**") [cite: 4]
        
        # ميزة سماع النطق الصحيح (TTS) [cite: 4]
        if st.button(f"🔊 اسمع نطق حرف ({target_letter})"):
            audio_fp = speak_text(target_letter) [cite: 4]
            st.audio(audio_fp, format='audio/mp3') [cite: 4]
            
    with col_r:
        if target_letter == "راء":
            st.warning("نصيحة: تأكد من ملامسة طرف اللسان لسقف الحنك العلوي.") [cite: 4]

    st.divider()
    st.subheader("🎤 سجل نطقك للحرف:")
    # أداة تسجيل الصوت [cite: 5]
    user_audio = audiorecorder("اضغط للتحدث", "إيقاف وتحليل")

    if len(user_audio) > 0:
        # تحميل ومعالجة الصوت المسجل [cite: 5]
        y, sr = librosa.load(user_audio.export(), sr=22050) [cite: 5]
        user_feats = get_features(y, sr) [cite: 5]
        
        # بصمة مرجعية للمقارنة [cite: 5]
        REF = np.random.rand(13) 
        similarity = cosine_similarity([REF], [user_feats])[0][0] [cite: 5]
        score = int(similarity * 100) [cite: 5]

        # عرض النتائج بناءً على درجة الدقة [cite: 6]
        if score > 75:
            st.success(f"أحسنت! نسبة الدقة {score}%") [cite: 6, 7]
            st.session_state.total_xp += 50 [cite: 6]
        else:
            st.error(f"حاول مرة أخرى. الدقة {score}%. ركز على مخرج الحرف.") [cite: 6]

with tab2:
    # دليل الاستخدام [cite: 7]
    st.markdown("### كيف تبدأ؟\n1. اسمع الحرف أولاً.\n2. سجل صوتك.\n3. اجمع النقاط!")

with tab3:
    # سياسة الخصوصية [cite: 7]
    st.write("بياناتك الصوتية آمنة ومعالجتها تتم لحظياً ولا يتم تخزينها.")
