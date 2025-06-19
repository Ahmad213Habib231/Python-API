# استخدم نسخة صغيرة من Python
FROM python:3.10-slim

# تثبيت dependencies الأساسية لنظام التشغيل
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# إعداد مجلد العمل
WORKDIR /app

# نسخ requirements فقط أولًا للاستفادة من cache
COPY requirements.txt .

# تثبيت البايثون باكجات بدون كاش لتقليل الحجم
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY . .

# تعيين البورت اللي هتشتغل عليه في Railway
ENV PORT=5000

# الأمر اللي يشغل التطبيق
CMD ["python", "app.py"]
