import re
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sklearn.linear_model import LogisticRegression

TOKEN = "8235101337:AAE07TjdyK_KoJQRVbc9nuSgYyPxGt638S8"

STATS_FILE = 'method_stats.json'
WINDOW_FOR_YOUR_METHOD = 50

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    stats = {
        'your_method': {'correct': 0, 'total': 0, 'score': 0},
        'logreg_method': {'correct': 0, 'total': 0, 'score': 0},
        'depth_method': {'correct': 0, 'total': 0, 'score': 0},
        'markov_method': {'correct': 0, 'total': 0, 'score': 0}
    }
    save_stats(stats)
    return stats

def save_stats(stats):
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)

def load_all_data():
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

def format_numbers(arr):
    return [int(x) for x in arr]

# ===================== ТВОЙ МЕТОД =====================
def your_full_method(data):
    if len(data) > WINDOW_FOR_YOUR_METHOD:
        data = data[-WINDOW_FOR_YOUR_METHOD:]
    
    variants = [[], [], []]
    
    for pos in range(6):
        history = data[:, pos]
        candidates = []
        
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
            
            if depth >= avg_int:
                candidates.append((depth, num))
        
        if not candidates:
            for num in range(1, 7):
                positions = [i for i, v in enumerate(history) if v == num]
                last_pos = max(positions) if positions else -1
                depth = len(data) - last_pos - 1
                candidates.append((depth, num))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        variants[0].append(candidates[0][1])
        
        if len(candidates) >= 2:
            variants[1].append(candidates[1][1])
        else:
            variants[1].append(candidates[0][1])
        
        if len(candidates) >= 3:
            variants[2].append(candidates[2][1])
        elif len(candidates) >= 2:
            variants[2].append(candidates[1][1])
        else:
            variants[2].append(candidates[0][1])
    
    return variants

# ===================== ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ =====================
def logreg_method(data):
    X_train = []
    y_train = []
    window = 10
    for i in range(window, len(data) - 1):
        features = []
        recent_sums = [sum(draw) for draw in data[i-window:i]]
        features.append(np.mean(recent_sums))
        for num in range(1, 7):
            positions = [j for j, draw in enumerate(data[:i]) if num in draw]
            depth = i - (max(positions) if positions else -1) - 1
            features.append(depth)
        X_train.append(features)
        y_train.append(data[i])
    if len(X_train) < 10:
        return [3, 3, 3, 4, 4, 4]
    combo = []
    for pos in range(6):
        y_pos = [draw[pos] for draw in y_train]
        model = LogisticRegression(max_iter=200, C=0.1)
        model.fit(X_train, y_pos)
        current_features = []
        recent_sums = [sum(draw) for draw in data[-window:]]
        current_features.append(np.mean(recent_sums))
        for num in range(1, 7):
            positions = [j for j, draw in enumerate(data) if num in draw]
            depth = len(data) - (max(positions) if positions else -1) - 1
            current_features.append(depth)
        pred = model.predict([current_features])[0]
        combo.append(pred)
    return combo

# ===================== ГЛУБИННЫЙ АНАЛИЗ =====================
def depth_method(data):
    combo = []
    for pos in range(6):
        history = data[:, pos]
        depths = {}
        for num in range(1, 7):
            positions = [i for i, v in enumerate(history) if v == num]
            last_pos = max(positions) if positions else -1
            depth = len(data) - last_pos - 1
            depths[num] = depth
        combo.append(max(depths, key=depths.get))
    return combo

# ===================== МАРКОВСКАЯ ЦЕПЬ =====================
def markov_method(data):
    combo = []
    for pos in range(6):
        transitions = defaultdict(Counter)
        for i in range(len(data) - 1):
            current = data[i, pos]
            next_val = data[i + 1, pos]
            transitions[current][next_val] += 1
        last = data[-1, pos]
        if transitions[last]:
            next_num = max(transitions[last], key=transitions[last].get)
        else:
            all_nums = [draw[pos] for draw in data]
            next_num = Counter(all_nums).most_common(1)[0][0]
        combo.append(next_num)
    return combo

def add_draw_to_file(numbers):
    with open('lottery.csv', 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    last_line = None
    for line in reversed(lines):
        if line.strip() and not line.startswith('lucky') and not line.startswith('﻿lucky'):
            if re.match(r'^\d+', line.strip()):
                last_line = line
                break
    
    if last_line:
        last_num = int(re.search(r'\d+', last_line).group())
    else:
        last_num = 0
    
    new_num = last_num + 1
    now = datetime.now()
    new_line = f'{new_num},{now.strftime("%d.%m.%y")}, {now.strftime("%H:%M")}, '
    for i, n in enumerate(numbers):
        new_line += str(n) + (', +' if i < 5 else '')
    new_line += '\n'
    
    lines.insert(-1, new_line)
    with open('lottery.csv', 'w', encoding='utf-8-sig') as f:
        f.writelines(lines)
    
    return new_num

async def start(update: Update, context):
    await update.message.reply_text(
        "🎰 *Лотерейный прогнозист* 🎰\n\n"
        "Команды:\n"
        "/predict - прогноз (4 метода)\n"
        "/add - добавить тираж\n"
        "/history - последние 5 тиражей\n"
        "/stats - статистика методов\n\n"
        f"📊 Твой метод использует последние {WINDOW_FOR_YOUR_METHOD} тиражей",
        parse_mode="Markdown"
    )

async def predict(update: Update, context):
    msg = await update.message.reply_text("🔄 Считаю прогнозы...")
    
    all_data = load_all_data()
    if len(all_data) < 10:
        await msg.edit_text("❌ Мало данных (нужно минимум 10 тиражей)")
        return
    
    stats = load_stats()
    
    your_variants = your_full_method(all_data)
    your_main = format_numbers(your_variants[0])
    your_alt1 = format_numbers(your_variants[1]) if len(your_variants) > 1 else your_main
    your_alt2 = format_numbers(your_variants[2]) if len(your_variants) > 2 else your_main
    
    logreg = format_numbers(logreg_method(all_data))
    depth = format_numbers(depth_method(all_data))
    markov = format_numbers(markov_method(all_data))
    
    best_method = None
    best_score = -1
    for name, stat in stats.items():
        if stat['total'] > 0 and stat['score'] > best_score:
            best_score = stat['score']
            best_method = name
    
    msg_text = "🔮 *ПРОГНОЗЫ НА СЛЕДУЮЩИЙ ТИРАЖ* 🔮\n\n"
    msg_text += "┌─────────────────────────────────┐\n"
    msg_text += f"│ 👑 *ВАШ МЕТОД (основной)*     │\n"
    msg_text += f"│    {your_main} │\n"
    msg_text += f"│    🎯 Сумма: *{sum(your_main)}*              │\n"
    msg_text += "├─────────────────────────────────┤\n"
    msg_text += f"│ 📌 *ВАШ МЕТОД (вар.2)*        │\n"
    msg_text += f"│    {your_alt1} │\n"
    msg_text += f"│    🎯 Сумма: *{sum(your_alt1)}*              │\n"
    msg_text += "├─────────────────────────────────┤\n"
    msg_text += f"│ 📌 *ВАШ МЕТОД (вар.3)*        │\n"
    msg_text += f"│    {your_alt2} │\n"
    msg_text += f"│    🎯 Сумма: *{sum(your_alt2)}*              │\n"
    msg_text += "├─────────────────────────────────┤\n"
    msg_text += f"│ 📊 *ЛОГ.РЕГРЕССИЯ*            │\n"
    msg_text += f"│    {logreg} │\n"
    msg_text += f"│    🎯 Сумма: *{sum(logreg)}*              │\n"
    msg_text += "├─────────────────────────────────┤\n"
    msg_text += f"│ 📈 *ГЛУБИННЫЙ АНАЛИЗ*         │\n"
    msg_text += f"│    {depth} │\n"
    msg_text += f"│    🎯 Сумма: *{sum(depth)}*              │\n"
    msg_text += "├─────────────────────────────────┤\n"
    msg_text += f"│ 🔄 *МАРКОВСКАЯ ЦЕПЬ*          │\n"
    msg_text += f"│    {markov} │\n"
    msg_text += f"│    🎯 Сумма: *{sum(markov)}*              │\n"
    msg_text += "└─────────────────────────────────┘\n"
    
    if best_method:
        name_rus = {
            'your_method': '👑 Ваш метод',
            'logreg_method': '📊 Логистическая регрессия',
            'depth_method': '📈 Глубинный анализ',
            'markov_method': '🔄 Марковская цепь'
        }[best_method]
        msg_text += f"\n⭐ *Рекомендуемый метод:* {name_rus} (точность {best_score:.0%})\n"
    else:
        msg_text += "\n📊 *Нет статистики. Добавьте тиражи через /add*"
    
    msg_text += f"\n📈 Всего в базе: {len(all_data)} тиражей"
    msg_text += f"\n📊 Твой метод использует последние {WINDOW_FOR_YOUR_METHOD} тиражей"
    
    await msg.edit_text(msg_text, parse_mode="Markdown")

async def history(update: Update, context):
    data = load_all_data()
    if len(data) == 0:
        await update.message.reply_text("❌ Нет данных")
        return
    
    with open('lottery.csv', 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    draws = []
    data_idx = 0
    for line in lines:
        if line.strip() and not line.startswith('lucky') and not line.startswith('﻿lucky'):
            match = re.match(r'(\d+)', line.strip())
            if match and data_idx < len(data):
                num = int(match.group(1))
                draws.append((num, data[data_idx]))
                data_idx += 1
    
    draws.sort(key=lambda x: x[0], reverse=True)
    last_5 = draws[:5]
    total = len(draws)
    
    msg = f"📋 *Последние 5 тиражей из {total}:*\n\n"
    for num, draw in last_5:
        num_str = f"{num:06d}"
        msg += f"┌─────────────────────────┐\n"
        msg += f"│ Тираж *{num_str}*\n"
        msg += f"│ {format_numbers(draw)}\n"
        msg += f"│ 🎯 Сумма: *{sum(draw)}*\n"
        msg += f"└─────────────────────────┘\n\n"
    
    await update.message.reply_text(msg, parse_mode="Markdown")

async def add(update: Update, context):
    await update.message.reply_text("📝 Введите 6 чисел через пробел, например:\n`2 3 4 3 5 4`", parse_mode="Markdown")
    context.user_data['waiting'] = True

async def stats(update: Update, context):
    stats = load_stats()
    
    msg = "📊 *СТАТИСТИКА МЕТОДОВ:*\n\n"
    msg += "┌─────────────────────────────────┐\n"
    
    for name, stat in stats.items():
        name_rus = {
            'your_method': '👑 Ваш метод',
            'logreg_method': '📊 Лог.регрессия',
            'depth_method': '📈 Глубинный анализ',
            'markov_method': '🔄 Марковская цепь'
        }[name]
        
        total = stat['total']
        correct = stat['correct']
        
        if total > 0:
            accuracy = (correct / total) * 100
            bar = "█" * int(accuracy / 5)
            msg += f"│ {name_rus:<25} │\n"
            msg += f"│    Точность: {accuracy:.0f}% {bar:<10} │\n"
            msg += f"│    Угадано: {correct}/{total} тиражей          │\n"
        else:
            msg += f"│ {name_rus:<25} │\n"
            msg += f"│    Нет данных              │\n"
        msg += "├─────────────────────────────────┤\n"
    
    msg += "└─────────────────────────────────┘\n"
    msg += "\n💡 *Совет:* Добавляйте тиражи через /add, чтобы накапливать статистику!"
    
    await update.message.reply_text(msg, parse_mode="Markdown")

async def handle_message(update: Update, context):
    if context.user_data.get('waiting'):
        try:
            nums = [int(x) for x in update.message.text.split()]
            if len(nums) == 6 and all(1 <= x <= 6 for x in nums):
                await update.message.reply_text("🔄 Добавляю тираж...")
                
                old_data = load_all_data()
                if len(old_data) >= 10:
                    old_variants = your_full_method(old_data)
                    old_your = old_variants[0]
                    old_logreg = logreg_method(old_data)
                    old_depth = depth_method(old_data)
                    old_markov = markov_method(old_data)
                    
                    stats = load_stats()
                    
                    for name, pred in [('your_method', old_your), ('logreg_method', old_logreg),
                                       ('depth_method', old_depth), ('markov_method', old_markov)]:
                        matches = len(set(pred) & set(nums))
                        correct = matches >= 3
                        stats[name]['total'] += 1
                        if correct:
                            stats[name]['correct'] += 1
                        if stats[name]['total'] > 0:
                            stats[name]['score'] = stats[name]['correct'] / stats[name]['total']
                        else:
                            stats[name]['score'] = 0
                    save_stats(stats)
                
                new_num = add_draw_to_file(nums)
                new_num_str = f"{new_num:06d}"
                await update.message.reply_text(f"✅ Тираж {new_num_str} добавлен!\n{nums} | сумма {sum(nums)}")
                context.user_data['waiting'] = False
            else:
                await update.message.reply_text("❌ Нужно 6 чисел от 1 до 6")
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print(f"✅ Бот запущен!")
    print(f"📊 Твой метод использует последние {WINDOW_FOR_YOUR_METHOD} тиражей")
    app.run_polling()

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print(f"🔄 Перезапуск через 10 секунд...")
            time.sleep(10)
