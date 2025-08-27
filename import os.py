import os
os.makedirs("E:\\huggingface_cache", exist_ok=True)
os.environ["HF_HOME"] = "E:\\huggingface_cache"
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import re

# ----------------------------
# فایل‌های ذخیره‌سازی
# ----------------------------
history_file = "sentiment_history.csv"
memory_file = "conversation_memory.json"

# بارگذاری یا ایجاد فایل تاریخچه
try:
    df = pd.read_csv(history_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Text", "Emotion", "Response"])
    df.to_csv(history_file, index=False)

# بارگذاری یا ایجاد حافظه مکالمه
try:
    with open(memory_file, "r", encoding="utf-8") as f:
        conversation_memory = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    conversation_memory = []

# ----------------------------
# رنگ‌ها برای احساسات
# ----------------------------
color_map = {
    "خوشحال": "#A3E4D7",
    "ناراحت": "#F1948A",
    "عصبی": "#F7DC6F",
    "خنثی": "#D7DBDD"
}

# ----------------------------
# نگاشت احساسات برای نمودار
# ----------------------------
emotion_to_english = {
    "خوشحال": "Happy",
    "ناراحت": "Sad",
    "عصبی": "Angry",
    "خنثی": "Neutral"
}

# ----------------------------
# تشخیص احساسات
# ----------------------------
def get_emotion(text):
    text_lower = text.lower()

    positive_words_fa = ["خوش", "شاد", "عالی", "خوب", "شادی", "لذت", "زیبا", "خوشحال","خوشبخت","راضی","حال خوب","سرحال","پرانرژی"]
    negative_words_fa = ["ناراحت", "غمگین", "بد", "غم", "اشتباه", "درد", "مشکل", "کلافه","خسته","ناامید","ناامید ام","دلخور","سوگوار","پریشون","بی حوصله"]
    angry_words_fa = ["عصبانی", "خشم", "اعصاب", "حرص", "عصبانیت", "ناراحتی","عصب","عصبانی تر","اعصاب","خشمگین","انتقام","نفرت","دلخور","شدید عصبانی","اشفته","رد دادم"]

    positive_words_en = ["happy", "good", "great", "joy", "glad", "awesome", "thankful", "love"]
    negative_words_en = ["sad", "bad", "unhappy", "angry", "pain", "problem", "upset", "frustrated"]
    angry_words_en = ["angry", "furious", "mad", "irritated", "annoyed"]

    if is_persian(text):
        positive_words = positive_words_fa
        negative_words = negative_words_fa
        angry_words = angry_words_fa
    else:
        positive_words = positive_words_en
        negative_words = negative_words_en
        angry_words = angry_words_en

    pos_count = sum(word in text_lower for word in positive_words)
    neg_count = sum(word in text_lower for word in negative_words)
    angry_count = sum(word in text_lower for word in angry_words)

    if angry_count > 0:
        return "عصبی"
    elif neg_count > 0:
        return "ناراحت"
    elif pos_count > 0:
        return "خوشحال"
    else:
        return "خنثی"

# ----------------------------
# تشخیص فارسی
# ----------------------------
def is_persian(text):
    return any('\u0600' <= ch <= '\u06FF' for ch in text)

# ----------------------------
# پاسخ‌های محاوره‌ای
# ----------------------------
responses_dict_fa = {
    "خوشحال": {
        "starts": ["😊  چه عالی!رفیق", "😃 خوشحالم برات!ای جان", "👍 امروز فوق‌العاده‌ست!برای تو", "🌟 خبر خوبیه!"],
        "ends": ["روزت عالی باشه!", "این انرژی مثبت ادامه داشته باشه!", "لذت ببر!", "همیشه خوشحال باشی!"]
    },
    "ناراحت": {
        "starts": ["💛 نگران نباش، همه چیز درست میشه.", "💪 من کنارتم.", "😔 می‌فهمم حالت رو.", "😢 می‌دونم ناراحتی.","پیشنهاد میکنم یه مدیتیشن کنی رفیق .",
                   "تو خیلی قوی تر از این حرفایی ها میدونستی؟" ],
        "ends": ["همه چیز بهتر میشه.", "سعی کن کمی استراحت کنی.", "من اینجام کنارت.", "قوی باش."]
    },
    "عصبی": {
        "starts": ["🧘‍♂ یه نفس عمیق بکش.", "🕊 سعی کن آروم باشی.", "😤 می‌فهمم حالت رو.", "💆‍♂ کمی استراحت کن."],
        "ends": ["حواست به خودت باشه.", "به خودت فشار نیار.", "همه چی درست میشه.", "یه لحظه آرام باش.","درکت میکنم رفیق من کنارتم."
                 ,"پیشنهاد میکنم یه مدیتیشن کنی رفیق .","تو خیلی قوی تر از این حرفایی ها میدونستی؟"]
    },
    "خنثی": {
        "starts": ["😐 فهمیدم.", "🤔 متوجه شدم.", "🙂 خب.", "😶 اوکی."],
        "ends": ["ادامه بده.", "متوجه شدم.", "اوکی، باشه.", "فهمیدم."]
    }
}

responses_dict_en = {
    "خوشحال": {
        "starts": ["😊 That's great!", "😃 I'm happy for you!", "👍 Awesome!", "🌟 Good news!"],
        "ends": ["Have a wonderful day!", "Keep this positive energy!", "Enjoy!", "Stay happy!"]
    },
    "ناراحت": {
        "starts": ["💛 Don't worry, it'll get better.", "💪 I'm here for you.", "😔 I understand how you feel.", "😢 I know it's tough."],
        "ends": ["Everything will improve.", "Try to take a break.", "I'm here with you.", "Stay strong."]
    },
    "عصبی": {
        "starts": ["🧘‍♂ Take a deep breath.", "🕊 Try to stay calm.", "😤 I get it, you're frustrated.", "💆‍♂ Take a short rest."],
        "ends": ["Take care of yourself.", "Don't pressure yourself.", "Everything will be fine.", "Relax for a moment."]
    },
    "خنثی": {
        "starts": ["😐 I see.", "🤔 Got it.", "🙂 Okay.", "😶 Alright."],
        "ends": ["Carry on.", "Understood.", "Okay then.", "Noted."]
    }
}

# ----------------------------
# تولید پاسخ محاوره‌ای
# ----------------------------
def generate_response(user_input):
    try:
        emotion = get_emotion(user_input)
        
        if is_persian(user_input):
            parts = responses_dict_fa[emotion]
        else:
            parts = responses_dict_en[emotion]
        
        start = random.choice(parts["starts"])
        end = random.choice(parts["ends"])
        
        response_text = f"{start} {end}".strip()
        return response_text
    except Exception as e:
        return f"خطا در تولید پاسخ: {e}"

# ----------------------------
# تشخیص احساس و نمایش پاسخ
# ----------------------------
def detect_sentiment():
    try:
        user_text = text_entry.get().strip()
        if not user_text:
            return

        text_entry.delete(0, tk.END)

        emotion = get_emotion(user_text)
        response_text = generate_response(user_text)

        conversation_text.insert(tk.END, f"شما: {user_text}\n")
        conversation_text.insert(tk.END, f"چت‌بات: {response_text}\n\n")
        conversation_text.see(tk.END)

        if is_persian(user_text):
            root.config(bg=color_map.get(emotion, "#D7DBDD"))
        else:
            root.config(bg="#D7DBDD")

        global df, conversation_memory

        # ذخیره در تاریخچه CSV
        df = pd.concat([df, pd.DataFrame({"Text": [user_text], "Emotion": [emotion], "Response": [response_text]})], ignore_index=True)

        # ذخیره در حافظه مکالمه JSON
        conversation_memory.append({
            "user": user_text,
            "emotion": emotion,
            "assistant": response_text
        })

    except Exception as e:
        conversation_text.insert(tk.END, f"خطا: {str(e)}\n\n")
        print(f"Error in detect_sentiment: {e}")

# ----------------------------
# ذخیره داده‌ها در زمان خروج
# ----------------------------
def save_data_on_exit():
    try:
        df.to_csv(history_file, index=False)
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(conversation_memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving data: {e}")

# ----------------------------
# نمودار احساسات
# ----------------------------
def show_chart():
    try:
        if df.empty:
            conversation_text.insert(tk.END, "No data available to display the chart.\n\n")
            return
        counts = df['Emotion'].map(emotion_to_english).value_counts()
        counts.plot(kind='bar', color=[color_map.get(e, '#D7DBDD') for e in df['Emotion']])
        plt.title("Sentiment Distribution")
        plt.ylabel("Count")
        plt.xlabel("Sentiment")
        plt.show()
    except Exception as e:
        conversation_text.insert(tk.END, f"Error displaying chart: {str(e)}\n\n")
        print(f"Error in show_chart: {e}")

# ----------------------------
# GUI
# ----------------------------
# GUI
# ----------------------------
root = tk.Tk()
root.title("English & Persian EmotiBot")
root.geometry("750x650")
root.config(bg="#f0f4f8")

# ----------------------------
# تغییر رنگ نرم پس‌زمینه
# ----------------------------
def animate_bg(target_color, steps=20, delay=20):
    # تبدیل رنگ hex به RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2 ,4))
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    start_rgb = hex_to_rgb(root['bg'])
    end_rgb = hex_to_rgb(target_color)
    
    delta = [(e - s)/steps for s, e in zip(start_rgb, end_rgb)]
    
    for i in range(steps):
        new_rgb = [int(start_rgb[j] + delta[j]*i) for j in range(3)]
        root.config(bg=rgb_to_hex(new_rgb))
        root.update()
        root.after(delay)

# ----------------------------
# Label
# ----------------------------
label = tk.Label(root, text="Enter your sentence (Persian or English):",
         bg="#f0f4f8", font=("Helvetica", 13, "bold"), fg="#333333")
label.pack(pady=(10,5))

# ----------------------------
# Entry
# ----------------------------
text_entry = tk.Entry(root, width=80, font=("Helvetica", 12),
                      bg="#ffffff", fg="#333333", bd=2, relief="groove")
text_entry.pack(pady=(0,10))
root.bind('<Return>', lambda event: detect_sentiment())

# ----------------------------
# Button hover effect
# ----------------------------
def on_enter(e):
    e.widget['background'] = e.widget.hover_bg

def on_leave(e):
    e.widget['background'] = e.widget.default_bg

# ----------------------------
# Frame برای دکمه‌ها
# ----------------------------
button_frame = tk.Frame(root, bg="#f0f4f8")
button_frame.pack(pady=(0,10))

send_btn = tk.Button(button_frame, text="Send Sentence & Get Response", command=detect_sentiment,
          font=("Helvetica", 12), bg="#4CAF50", fg="white", bd=0, padx=10, pady=5)
send_btn.pack(side=tk.LEFT, padx=5)
send_btn.default_bg = "#4CAF50"
send_btn.hover_bg = "#45a049"
send_btn.bind("<Enter>", on_enter)
send_btn.bind("<Leave>", on_leave)

chart_btn = tk.Button(button_frame, text="Show Sentiment Chart", command=show_chart,
          font=("Helvetica", 12), bg="#2196F3", fg="white", bd=0, padx=10, pady=5)
chart_btn.pack(side=tk.LEFT, padx=5)
chart_btn.default_bg = "#2196F3"
chart_btn.hover_bg = "#1976D2"
chart_btn.bind("<Enter>", on_enter)
chart_btn.bind("<Leave>", on_leave)

# ----------------------------
# Frame برای مکالمه با Scrollbar
# ----------------------------
conversation_frame = tk.Frame(root, bg="#f0f4f8")
conversation_frame.pack(pady=5)

scrollbar = tk.Scrollbar(conversation_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

conversation_text = tk.Text(conversation_frame, width=85, height=25,
                            font=("Helvetica", 11), yscrollcommand=scrollbar.set,
                            bg="#ffffff", fg="#333333", bd=2, relief="groove")
conversation_text.pack()
scrollbar.config(command=conversation_text.yview)

# ----------------------------
# تغییر رنگ پس‌زمینه بر اساس احساس
# ----------------------------
def update_bg_by_emotion(emotion):
    color_map = {
        "خوشحال": "#A3E4D7",
        "ناراحت": "#F1948A",
        "عصبی": "#F7DC6F",
        "خنثی": "#D7DBDD"
    }
    target_color = color_map.get(emotion, "#D7DBDD")
    animate_bg(target_color)

# ----------------------------
# highlight کوتاه برای جمله
# ----------------------------
def highlight_last_text():
    conversation_text.tag_add("highlight", "end-2l", "end-1c")
    conversation_text.tag_config("highlight", background="#ffff99")
    root.after(500, lambda: conversation_text.tag_delete("highlight"))

# ----------------------------
# بازنویسی detect_sentiment برای GUI جدید
# ----------------------------
def detect_sentiment_gui():
    user_text = text_entry.get().strip()
    if not user_text:
        return
    text_entry.delete(0, tk.END)
    emotion = get_emotion(user_text)
    response_text = generate_response(user_text)

    conversation_text.insert(tk.END, f"شما: {user_text}\n")
    conversation_text.insert(tk.END, f"چت‌بات: {response_text}\n\n")
    conversation_text.see(tk.END)
    
    update_bg_by_emotion(emotion)
    highlight_last_text()

    global df, conversation_memory
    df = pd.concat([df, pd.DataFrame({"Text": [user_text], "Emotion": [emotion], "Response": [response_text]})], ignore_index=True)
    conversation_memory.append({
        "user": user_text,
        "emotion": emotion,
        "assistant": response_text
    })

# جایگزین bind و دکمه با GUI جدید
root.bind('<Return>', lambda event: detect_sentiment_gui())
send_btn.config(command=detect_sentiment_gui)

# ----------------------------
# ذخیره داده‌ها در خروج
# ----------------------------
root.protocol("WM_DELETE_WINDOW", save_data_on_exit)

print("برنامه آماده اجرا است.")
root.mainloop()