import re
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os

# ---------------------- КОНФИГУРАЦИЯ ----------------------
TOKEN = "8235101337:AAE07TjdyK_KoJQRVbc9nuSgYyPxGt638S8"

# ---------------------- ЗАГРУЗКА ДАННЫХ ----------------------
def load_data():
    with open('lottery.csv', 'r', encoding='utf-8-sig') as f:
        text = f.read()
    all_numbers = [int(n) for n in re.findall(r'\d+', text)]
    data = []
    i = 0
    while i < len(all_numbers):
        if i + 11 < len(all_numbers):
            candidate = all_numbers[i+5:i+11]
            if all(1 <= x <= 6 for x in candidate):
                data.append(candidate)
                i += 11
            else:
                i += 1
        else:
            break
    return np.array(data)

# ---------------------- МЕТОД ПРОГНОЗА (ВАШ) ----------------------
def get_prediction(data):
    """Прогноз на основе глубины"""
    prediction = []
    
    for pos in range(6):
        history = data[:, pos]
        scores = {}
        
        for num in range(1, 7):
            positions = [i for i, v in enumerate(history) if v == num]
            
            if len(positions) > 1:
                intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                avg_int = np.mean(intervals)
            elif len(positions) == 1:
                avg_int = len(data) - positions[0]
            else:
                avg_int = len(data) + 10
            
            last_pos = max(positions) if positions else -1
            depth = len(data) - last_pos - 1
            ratio = depth / avg_int if avg_int > 0 else 0
            
            if ratio >= 2.0:
                prob = 1.0
            elif ratio >= 1.5:
                prob = 0.9
            elif ratio >= 1.2:
                prob = 0.7
            elif ratio >= 0.8:
                prob = 0.6
            elif ratio >= 0.5:
                prob = 0.4
            else:
                prob = 0.2
            
            scores[num] = prob
        
        best = max(scores, key=scores.get)
        prediction.append(best)
    
    return prediction

# ---------------------- КОМАНДЫ БОТА ----------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎰 *Лотерейный прогнозист* 🎰\n\n"
        "Доступные команды:\n"
        "/history - последние 5 тиражей\n"
        "/predict - прогноз на следующий тираж\n"
        "/add - добавить прошедший тираж\n"
        "/stats - статистика\n\n"
        "📊 Прогноз основан на анализе глубины чисел!",
        parse_mode="Markdown"
    )

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = load_data()
    if len(data) == 0:
        await update.message.reply_text("❌ Нет данных в файле!")
        return
    
    total = len(data)
    last_5 = data[-5:]
    
    message = f"📋 *Последние 5 тиражей из {total}:*\n\n"
    for i, draw in enumerate(last_5):
        draw_num = total - 5 + i + 1
        message += f"{draw_num}: {list(draw)} | Сумма: {sum(draw)}\n"
    
    await update.message.reply_text(message, parse_mode="Markdown")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Анализирую историю, подождите...")
    
    data = load_data()
    if len(data) < 10:
        await update.message.reply_text("❌ Недостаточно данных для прогноза (нужно минимум 10 тиражей)")
        return
    
    prediction = get_prediction(data)
    pred_sum = sum(prediction)
    
    message = f"🔮 *Прогноз на следующий тираж:*\n\n"
    message += f"🎯 Комбинация: {prediction}\n"
    message += f"📊 Сумма: {pred_sum}\n\n"
    
    if pred_sum <= 14:
        message += "⚠️ ЭКСТРЕМАЛЬНО НИЗКАЯ СУММА!"
    elif pred_sum >= 28:
        message += "⚠️ ЭКСТРЕМАЛЬНО ВЫСОКАЯ СУММА!"
    else:
        message += "✅ Средняя сумма"
    
    await update.message.reply_text(message, parse_mode="Markdown")

async def add_draw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📝 Введите 6 чисел через пробел\n"
        "Пример: `2 3 4 3 5 4`\n\n"
        "Каждое число от 1 до 6",
        parse_mode="Markdown"
    )
    context.user_data['waiting_for_numbers'] = True

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('waiting_for_numbers'):
        try:
            numbers = [int(x) for x in update.message.text.split()]
            if len(numbers) == 6 and all(1 <= x <= 6 for x in numbers):
                # Добавляем тираж в файл
                with open('lottery.csv', 'r', encoding='utf-8-sig') as f:
                    lines = f.readlines()
                
                last_line = None
                for line in reversed(lines):
                    if line.strip() and not line.startswith('lucky'):
                        last_line = line
                        break
                
                last_num = int(re.search(r'\d+', last_line).group()) if last_line else 0
                new_num = last_num + 1
                now = datetime.now()
                
                new_line = f'{new_num},{now.strftime("%d.%m.%y")}, {now.strftime("%H:%M")}, '
                for i, n in enumerate(numbers):
                    new_line += str(n) + (', +' if i < 5 else '')
                new_line += '\n'
                
                lines.insert(-1, new_line)
                with open('lottery.csv', 'w', encoding='utf-8-sig') as f:
                    f.writelines(lines)
                
                await update.message.reply_text(f"✅ Тираж №{new_num} добавлен!\n\nКомбинация: {numbers}\nСумма: {sum(numbers)}")
                context.user_data['waiting_for_numbers'] = False
            else:
                await update.message.reply_text("❌ Ошибка! Нужно 6 чисел от 1 до 6. Попробуйте снова.")
        except:
            await update.message.reply_text("❌ Ошибка ввода! Пример: `2 3 4 3 5 4`", parse_mode="Markdown")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = load_data()
    if len(data) == 0:
        await update.message.reply_text("❌ Нет данных!")
        return
    
    all_nums = [num for draw in data for num in draw]
    counts = Counter(all_nums)
    total = len(all_nums)
    
    message = f"📊 *Статистика по {len(data)} тиражам:*\n\n"
    for num in range(1, 7):
        pct = counts[num] / total * 100
        bar = "█" * int(pct)
        message += f"Число {num}: {counts[num]} раз ({pct:.1f}%) {bar}\n"
    
    sums = [sum(draw) for draw in data]
    avg_sum = np.mean(sums)
    message += f"\n📈 Средняя сумма: {avg_sum:.1f}"
    
    await update.message.reply_text(message, parse_mode="Markdown")

# ---------------------- ЗАПУСК БОТА ----------------------
def main():
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("add", add_draw))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🤖 Бот запущен! Напишите ему /start в Telegram")
    app.run_polling()

if __name__ == "__main__":
    main()