import re
import numpy as np
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = "8235101337:AAE07TjdyK_KoJQRVbc9nuSgYyPxGt638S8"

# ===================== ЗАГРУЗКА ДАННЫХ =====================
def load_data():
    try:
        with open('lottery.csv', 'r', encoding='utf-8-sig') as f:
            text = f.read()
    except FileNotFoundError:
        return []
    
    numbers = [int(x) for x in re.findall(r'\d+', text)]
    data = []
    i = 0
    while i < len(numbers):
        if i + 11 < len(numbers):
            cand = numbers[i+5:i+11]
            if all(1 <= x <= 6 for x in cand):
                data.append(cand)
                i += 11
            else:
                i += 1
        else:
            break
    return data

# ===================== ТВОЙ МЕТОД =====================
def get_prediction(data):
    if len(data) > 50:
        data = data[-50:]
    
    prediction = []
    for pos in range(6):
        history = [d[pos] for d in data]
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

# ===================== ДОБАВЛЕНИЕ ТИРАЖА =====================
def add_draw(numbers):
    # Читаем текущие данные
    draws = load_data()
    new_num = len(draws) + 1
    now = datetime.now()
    
    # Формируем строку
    new_line = f'{new_num},{now.strftime("%d.%m.%y")}, {now.strftime("%H:%M")}, '
    for i, n in enumerate(numbers):
        new_line += str(n) + (', +' if i < 5 else '')
    new_line += '\n'
    
    # Добавляем в файл
    with open('lottery.csv', 'a', encoding='utf-8-sig') as f:
        f.write(new_line)
    
    return new_num

# ===================== КОМАНДЫ =====================
async def start(update, context):
    await update.message.reply_text("🎰 Бот запущен!\n/predict - прогноз\n/add - добавить тираж\n/history - история")

async def predict(update, context):
    data = load_data()
    if len(data) < 10:
        await update.message.reply_text("❌ Мало данных (нужно минимум 10 тиражей)")
        return
    pred = get_prediction(data)
    await update.message.reply_text(f"🔮 Прогноз: {pred}\nСумма: {sum(pred)}")

async def history(update, context):
    data = load_data()
    if not data:
        await update.message.reply_text("❌ Нет данных")
        return
    total = len(data)
    last5 = data[-5:]
    msg = f"📋 Последние 5 тиражей из {total}:\n\n"
    for i, d in enumerate(last5):
        num = total - 5 + i + 1
        msg += f"{num}: {d} | сумма {sum(d)}\n"
    await update.message.reply_text(msg)

async def add(update, context):
    await update.message.reply_text("📝 Введите 6 чисел через пробел (от 1 до 6)")
    context.user_data['waiting'] = True

async def handle(update, context):
    if context.user_data.get('waiting'):
        try:
            nums = [int(x) for x in update.message.text.split()]
            if len(nums) == 6 and all(1 <= x <= 6 for x in nums):
                new_num = add_draw(nums)
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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    print("✅ Бот запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()
