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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

TOKEN = "8235101337:AAE07TjdyK_KoJQRVbc9nuSgYyPxGt638S8"

# ===================== НАСТРОЙКИ =====================
DATA_DIR = '/app/data'  # Путь к Volume на Railway
os.makedirs(DATA_DIR, exist_ok=True)
STATS_FILE = os.path.join(DATA_DIR, 'method_stats.json')
CSV_FILE = os.path.join(DATA_DIR, 'lottery.csv')

WINDOW_FOR_YOUR_METHOD = 50
MAX_DRAWS_FOR_LSTM = 300

# ===================== ЗАГРУЗКА/СОХРАНЕНИЕ =====================
def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    stats = {
        'your_method': {'correct': 0, 'total': 0, 'score': 0},
        'logreg_method': {'correct': 0, 'total': 0, 'score': 0},
        'depth_method': {'correct': 0, 'total': 0, 'score': 0},
        'markov_method': {'correct': 0, 'total': 0, 'score': 0},
        'lstm_method': {'correct': 0, 'total': 0, 'score': 0}
    }
    save_stats(stats)
    return stats

def save_stats(stats):
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)

def load_all_data():
    """Надёжная загрузка всех тиражей из CSV"""
    if not os.path.exists(CSV_FILE):
        return np.array([])
    
    with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if 'lucky-numbers' in line.lower():
            continue
        # Ищем 6 чисел от 1 до 6
        numbers = re.findall(r'\b([1-6])\b', line)
        if len(numbers) == 6:
            data.append([int(n) for n in numbers])
    
    return np.array(data)

def load_data_for_lstm():
    data = load_all_data()
    if len(data) > MAX_DRAWS_FOR_LSTM:
        data = data[-MAX_DRAWS_FOR_LSTM:]
    return data

def format_numbers(arr):
    return [int(x) for x in arr]

def get_last_draw_number():
    """Возвращает последний номер тиража из CSV"""
    if not os.path.exists(CSV_FILE):
        return 0
    
    with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    last_num = 0
    for line in lines:
        match = re.match(r'(\d+)', line.strip())
        if match:
            last_num = max(last_num, int(match.group(1)))
    return last_num

def add_draw_to_file(numbers):
    """Надёжное добавление тиража без перемешивания"""
    new_num = get_last_draw_number() + 1
    now = datetime.now()
    
    # Формируем строку без лишних символов
    numbers_str = ','.join(str(n) for n in numbers)
    new_line = f"{new_num:06d},{now.strftime('%d.%m.%y')},{now.strftime('%H:%M')},{numbers_str}\n"
    
    # Читаем существующий файл
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Проверка на дубликат (последний тираж)
    if lines:
        last_line = lines[-1].strip()
        last_numbers = re.findall(r'\b([1-6])\b', last_line)
        if len(last_numbers) == 6 and [int(n) for n in last_numbers] == numbers:
            return None  # Дубликат
    
    # Вставляем перед lucky-numbers.ru или в конец
    lucky_index = -1
    for i, line in enumerate(lines):
        if 'lucky-numbers.ru' in line.lower():
            lucky_index = i
            break
    
    if lucky_index != -1:
        lines.insert(lucky_index, new_line)
    else:
        lines.append(new_line)
    
    with open(CSV_FILE, 'w', encoding='utf-8-sig') as f:
        f.writelines(lines)
    
    return new_num

# ===================== LSTM =====================
def lstm_method(data):
    if len(data) < 50:
        return [3, 3, 3, 4, 4, 4]
    window = 20
    X, y = [], []
    sums = np.sum(data, axis=1).reshape(-1, 1)
    sums_norm = (sums - 6) / 30
    for i in range(len(data) - window - 1):
        window_data = data[i:i+window]
        window_sums = sums_norm[i:i+window]
        combined = np.concatenate([window_data, window_sums], axis=1)
        X.append(combined)
        y.append(data[i+window])
    X = np.array(X)
    y = np.array(y)
    X[:, :, :6] = (X[:, :, :6] - 1) / 5.0
    y_onehot = [to_categorical(y[:, p] - 1, num_classes=6) for p in range(6)]
    y_comb = np.hstack(y_onehot)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_comb[:split], y_comb[split:]
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(window, 7)),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(36, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, callbacks=[early_stop], verbose=0)
    last_window = data[-window:]
    last_sums = sums_norm[-window:]
    X_last = np.concatenate([last_window, last_sums], axis=1).reshape(1, window, 7)
    X_last[:, :, :6] = (X_last[:, :, :6] - 1) / 5.0
    pred = model.predict(X_last, verbose=0).reshape(6, 6)
    return [int(np.argmax(pred[i]) + 1) for i in range(6)]

# ===================== ВАШ МЕТОД =====================
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

# ===================== ИСПРАВЛЕНИЕ ДАННЫХ =====================
async def fixdata(update: Update, context):
    """Принудительно записывает правильные данные в файл"""
    correct_data = """004439,10.04.26,17:00,2,5,1,2,3,4
004440,10.04.26,18:00,2,3,4,1,3,6
004441,10.04.26,19:00,2,1,3,6,5,3
004442,10.04.26,20:00,5,5,2,5,6,1
004443,10.04.26,21:00,6,4,2,2,3,6
004444,10.04.26,22:00,6,3,3,4,4,2"""
    
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Удаляем старые строки с 004439 по 004444
    new_lines = []
    for line in lines:
        if not re.match(r'00443[9]|00444[0-4]', line):
            new_lines.append(line)
    
    # Добавляем правильные данные
    correct_lines = [line + '\n' for line in correct_data.split('\n')]
    
    lucky_index = -1
    for i, line in enumerate(new_lines):
        if 'lucky-numbers.ru' in line.lower():
            lucky_index = i
            break
    
    if lucky_index != -1:
        for i, corr_line in enumerate(correct_lines):
            new_lines.insert(lucky_index + i, corr_line)
    else:
        new_lines.extend(correct_lines)
    
    with open(CSV_FILE, 'w', encoding='utf-8-sig') as f:
        f.writelines(new_lines)
    
    await update.message.reply_text("✅ Данные исправлены!")

# ===================== КОМАНДЫ БОТА =====================
async def start(update: Update, context):
    await update.message.reply_text(
        "🎰 *Лотерейный прогнозист* 🎰\n\n"
        "Команды:\n"
        "/predict - прогноз (4 метода + LSTM)\n"
        "/add - добавить тираж\n"
        "/history - последние 5 тиражей\n"
        "/stats - статистика методов\n"
        "/fixdata - исправить данные (если суммы не совпадают)\n\n"
        f"📊 Твой метод использует последние {WINDOW_FOR_YOUR_METHOD} тиражей\n"
        f"🧠 LSTM обучен на последних {MAX_DRAWS_FOR_LSTM} тиражах",
        parse_mode="Markdown"
    )

async def predict(update: Update, context):
    msg = await update.message.reply_text("🔄 Считаю прогнозы (LSTM 20-30 сек)...")
    
    all_data = load_all_data()
    if len(all_data) < 10:
        await msg.edit_text("❌ Мало данных (нужно минимум 10 тиражей)")
        return
    
    lstm_data = load_data_for_lstm()
    stats = load_stats()
    
    your_variants = your_full_method(all_data)
    your_main = format_numbers(your_variants[0])
    your_alt1 = format_numbers(your_variants[1]) if len(your_variants) > 1 else your_main
    your_alt2 = format_numbers(your_variants[2]) if len(your_variants) > 2 else your_main
    
    logreg = format_numbers(logreg_method(all_data))
    depth = format_numbers(depth_method(all_data))
    markov = format_numbers(markov_method(all_data))
    lstm = format_numbers(lstm_method(lstm_data))
    
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
    msg_text += "├─────────────────────────────────┤\n"
    msg_text += f"│ 🧠 *LSTM НЕЙРОСЕТЬ*           │\n"
    msg_text += f"│    {lstm} │\n"
    msg_text += f"│    🎯 Сумма: *{sum(lstm)}*              │\n"
    msg_text += "└─────────────────────────────────┘\n"
    
    if best_method:
        name_rus = {
            'your_method': '👑 Ваш метод',
            'logreg_method': '📊 Логистическая регрессия',
            'depth_method': '📈 Глубинный анализ',
            'markov_method': '🔄 Марковская цепь',
            'lstm_method': '🧠 LSTM нейросеть'
        }[best_method]
        msg_text += f"\n⭐ *Рекомендуемый метод:* {name_rus} (точность {best_score:.0%})\n"
    else:
        msg_text += "\n📊 *Нет статистики. Добавьте тиражи через /add*"
    
    msg_text += f"\n📈 Всего в базе: {len(all_data)} тиражей"
    msg_text += f"\n📊 Твой метод использует последние {WINDOW_FOR_YOUR_METHOD} тиражей"
    msg_text += f"\n🧠 LSTM обучен на последних {MAX_DRAWS_FOR_LSTM} тиражах"
    
    await msg.edit_text(msg_text, parse_mode="Markdown")

async def history(update: Update, context):
    data = load_all_data()
    if len(data) == 0:
        await update.message.reply_text("❌ Нет данных")
        return
    
    # Получаем последние 5 с номерами тиражей
    draws_with_nums = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        data_idx = 0
        for line in lines:
            line = line.strip()
            if not line or 'lucky-numbers' in line.lower():
                continue
            match = re.match(r'(\d+)', line)
            if match and data_idx < len(data):
                draw_num = int(match.group(1))
                draws_with_nums.append((draw_num, data[data_idx]))
                data_idx += 1
    
    draws_with_nums.sort(key=lambda x: x[0], reverse=True)
    last_5 = draws_with_nums[:5]
    total = len(data)
    
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
            'markov_method': '🔄 Марковская цепь',
            'lstm_method': '🧠 LSTM'
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
                    old_lstm = lstm_method(load_data_for_lstm())
                    
                    stats = load_stats()
                    
                    for name, pred in [('your_method', old_your), ('logreg_method', old_logreg),
                                       ('depth_method', old_depth), ('markov_method', old_markov),
                                       ('lstm_method', old_lstm)]:
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
                if new_num is None:
                    await update.message.reply_text("⚠️ Этот тираж уже есть в базе!")
                else:
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
    app.add_handler(CommandHandler("fixdata", fixdata))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print(f"✅ Бот запущен!")
    print(f"📁 Файлы хранятся в Volume: {DATA_DIR}")
    print(f"📊 Твой метод использует последние {WINDOW_FOR_YOUR_METHOD} тиражей")
    print(f"🧠 LSTM обучен на последних {MAX_DRAWS_FOR_LSTM} тиражах")
    app.run_polling()

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print(f"🔄 Перезапуск через 10 секунд...")
            time.sleep(10)
