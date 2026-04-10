import os
import re
import time
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
import json
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ===================== ПУТЬ К ФАЙЛУ =====================
DATA_FILE = '/app/data/lottery.csv'
TOKEN = "8235101337:AAE07TjdyK_KoJQRVbc9nuSgYyPxGt638S8"

# ===================== ПРОСТАЯ ЗАГРУЗКА =====================
def load_draws():
    """Загружает тиражи из файла"""
    if not os.path.exists(DATA_FILE):
        return []
    draws = []
    with open(DATA_FILE, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('lucky') or line.startswith('Номер'):
                continue
            # Ищем номер тиража и числа
            parts = re.split(r'[,+]', line)
            nums = []
            for p in parts:
                p = p.strip()
                if p.isdigit() and 1 <= int(p) <= 6:
                    nums.append(int(p))
            if len(nums) == 6:
                draws.append(nums)
    return draws

def save_draw(draw):
    """Добавляет тираж в файл"""
    draws = load_draws()
    new_num = len(draws) + 1
    now = datetime.now()
    # Формируем строку
    line = f'{new_num},{now.strftime("%d.%m.%y")}, {now.strftime("%H:%M")}, '
    for i, n in enumerate(draw):
        line += str(n) + (' , +' if i < 5 else '')
    line += '\n'
    
    with open(DATA_FILE, 'a', encoding='utf-8-sig') as f:
        f.write(line)
    return new_num

def get_prediction():
    """Твой метод (упрощённо)"""
    draws = load_draws()
    if len(draws) < 10:
        return None
    # Берём последние 50
    if len(draws) > 50:
        draws = draws[-50:]
    
    prediction = []
    for pos in range(6):
        history = [d[pos] for d in draws]
        best_num = 1
        best_depth = -1
        for num in range(1, 7):
            positions = [i for i, v in enumerate(history) if v == num]
            last = max(positions) if positions else -1
            depth = len(history) - last - 1
            if depth > best_depth:
                best_depth = depth
                best_num = num
        prediction.append(best_num)
    return prediction

# ===================== КОМАНДЫ =====================
async def start(update: Update, context):
    await update.message.reply_text(
        "🎰 *Лотерейный прогнозист* 🎰\n\n"
        "/predict - прогноз\n"
        "/add - добавить тираж\n"
        "/history - последние 5 тиражей\n"
        "/stats - статистика\n"
        f"\n📁 Файл: {DATA_FILE}",
        parse_mode="Markdown"
    )

async def predict(update: Update, context):
    msg = await update.message.reply_text("🔄 Считаю...")
    pred = get_prediction()
    if pred is None:
        await msg.edit_text("❌ Мало данных (нужно минимум 10 тиражей)")
        return
    await msg.edit_text(f"🔮 Прогноз: {pred}\nСумма: {sum(pred)}")

async def history(update: Update, context):
    draws = load_draws()
    if not draws:
        await update.message.reply_text("❌ Нет данных")
        return
    total = len(draws)
    last5 = draws[-5:]
    msg = f"📋 *Последние 5 тиражей из {total}:*\n\n"
    for i, d in enumerate(last5):
        num = total - 5 + i + 1
        msg += f"{num}: {d} | сумма {sum(d)}\n"
    await update.message.reply_text(msg, parse_mode="Markdown")

async def add(update: Update, context):
    await update.message.reply_text("📝 Введите 6 чисел через пробел")
    context.user_data['waiting'] = True

async def stats(update: Update, context):
    draws = load_draws()
    if not draws:
        await update.message.reply_text("❌ Нет данных")
        return
    all_nums = [n for d in draws for n in d]
    counts = Counter(all_nums)
    msg = "📊 *Статистика:*\n\n"
    for num in range(1, 7):
        cnt = counts.get(num, 0)
        pct = cnt / len(all_nums) * 100
        msg += f"{num}: {cnt} раз ({pct:.1f}%)\n"
    sums = [sum(d) for d in draws]
    msg += f"\nСредняя сумма: {np.mean(sums):.1f}"
    await update.message.reply_text(msg, parse_mode="Markdown")

async def upload(update: Update, context):
    await update.message.reply_text("📁 Отправьте файл lottery.csv")

async def handle_document(update: Update, context):
    doc = update.message.document
    if doc.file_name == 'lottery.csv':
        await update.message.reply_text("🔄 Загружаю...")
        file = await doc.get_file()
        await file.download_to_drive(DATA_FILE)
        await update.message.reply_text(f"✅ Загружен в {DATA_FILE}")
    else:
        await update.message.reply_text("❌ Нужен lottery.csv")

async def handle_message(update: Update, context):
    if context.user_data.get('waiting'):
        try:
            nums = [int(x) for x in update.message.text.split()]
            if len(nums) == 6 and all(1 <= x <= 6 for x in nums):
                new_num = save_draw(nums)
                await update.message.reply_text(f"✅ Тираж {new_num} добавлен!\n{nums} | сумма {sum(nums)}")
                context.user_data['waiting'] = False
            else:
                await update.message.reply_text("❌ Нужно 6 чисел от 1 до 6")
        except:
            await update.message.reply_text("❌ Ошибка")

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("upload", upload))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Бот запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()
