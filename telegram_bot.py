import os
import re
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import json
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ===================== ПУТЬ К ФАЙЛУ =====================
DATA_FILE = 'lottery.csv'
TOKEN = "8235101337:AAE07TjdyK_KoJQRVbc9nuSgYyPxGt638S8"

STATS_FILE = 'method_stats.json'
MAX_DRAWS_FOR_LSTM = 300
WINDOW_FOR_YOUR_METHOD = 50

# ===================== ИСПРАВЛЕННАЯ ЗАГРУЗКА ДАННЫХ =====================
def load_all_data():
    """Читает файл в формате: 004444,10.04.26, 22:00,6,+,3,+,3,+,4,+,4,+,2"""
    if not os.path.exists(DATA_FILE):
        return np.array([])
    with open(DATA_FILE, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('lucky'):
            continue
        # Находим все числа в строке
        numbers = re.findall(r'\b\d+\b', line)
        # Берём последние 6 чисел (это шары)
        if len(numbers) >= 6:
            balls = [int(n) for n in numbers[-6:]]
            if all(1 <= x <= 6 for x in balls):
                data.append(balls)
    return np.array(data)

def load_data_for_lstm():
    data = load_all_data()
    if len(data) > MAX_DRAWS_FOR_LSTM:
        data = data[-MAX_DRAWS_FOR_LSTM:]
    return data

def format_numbers(arr):
    return [int(x) for x in arr]

# ===================== СТАТИСТИКА =====================
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
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', encoding='utf-8-sig') as f:
            f.write("")
    
    # Читаем существующие строки
    with open(DATA_FILE, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    # Определяем новый номер тиража
    max_num = 0
    for line in lines:
        line = line.strip()
        if line and not line.startswith('lucky'):
            match = re.match(r'(\d+)', line)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
    
    new_num = max_num + 1
    now = datetime.now()
    new_line = f'{new_num},{now.strftime("%d.%m.%y")}, {now.strftime("%H:%M")}, '
    for i, n in enumerate(numbers):
        new_line += str(n) + (', +' if i < 5 else '')
    new_line += '\n'
    
    lines.append(new_line)
    
    with open(DATA_FILE, 'w', encoding='utf-8-sig') as f:
        f.writelines(lines)
        f.flush()
    
    return new_num

def delete_last_draw():
    if not os.path.exists(DATA_FILE):
        return False, "Файл не найден"
    
    with open(DATA_FILE, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        return False, "Нет тиражей"
    
    deleted_line = lines[-1].strip()
    lines = lines[:-1]
    
    with open(DATA_FILE, 'w', encoding='utf-8-sig') as f:
        f.writelines(lines)
        f.flush()
    
    return True, deleted_line

async def start(update: Update, context):
    await update.message.reply_text(
        "🎰 *Лотерейный прогнозист* 🎰\n\n"
        "Команды:\n"
        "/predict - прогноз (5 методов)\n"
        "/add - добавить тираж\n"
        "/del - удалить последний тираж\n"
        "/history - последние 5 тиражей\n"
        "/stats - статистика методов\n"
        "/upload - загрузить файл lottery.csv\n\n"
        f"📁 Файл: {DATA_FILE}",
        parse_mode="Markdown"
    )

async def predict(update: Update, context):
    msg = await update.message.reply_text("🔄 Считаю прогнозы...")
    
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
    
    msg_text = "🔮 *ПРОГНОЗЫ:*\n\n"
    msg_text += f"👑 ВАШ МЕТОД (осн): {your_main} | сумма {sum(your_main)}\n"
    msg_text += f"📌 ВАШ МЕТОД (вар.2): {your_alt1} | сумма {sum(your_alt1)}\n"
    msg_text += f"📌 ВАШ МЕТОД (вар.3): {your_alt2} | сумма {sum(your_alt2)}\n"
    msg_text += f"📊 ЛОГ.РЕГРЕССИЯ: {logreg} | сумма {sum(logreg)}\n"
    msg_text += f"📈 ГЛУБИННЫЙ АНАЛИЗ: {depth} | сумма {sum(depth)}\n"
    msg_text += f"🔄 МАРКОВСКАЯ ЦЕПЬ: {markov} | сумма {sum(markov)}\n"
    msg_text += f"🧠 LSTM: {lstm} | сумма {sum(lstm)}\n"
    
    await msg.edit_text(msg_text, parse_mode="Markdown")

async def history(update: Update, context):
    data = load_all_data()
    if len(data) == 0:
        await update.message.reply_text("❌ Нет данных")
        return
    
    total = len(data)
    last_5 = data[-5:]
    
    # Определяем начальный номер (если файл начинается с 004444, то номер = 4444 - 5 + i)
    # Но проще показывать без номеров, только комбинации
    msg = f"📋 *Последние 5 тиражей из {total}:*\n\n"
    for i, draw in enumerate(last_5):
        msg += f"{draw} | сумма {sum(draw)}\n"
    
    await update.message.reply_text(msg, parse_mode="Markdown")

async def add(update: Update, context):
    await update.message.reply_text("📝 Введите 6 чисел через пробел, например:\n`2 3 4 3 5 4`", parse_mode="Markdown")
    context.user_data['waiting'] = True

async def delete_last(update: Update, context):
    success, result = delete_last_draw()
    if success:
        await update.message.reply_text(f"✅ Удалён тираж:\n`{result}`", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"❌ {result}")

async def stats(update: Update, context):
    stats = load_stats()
    msg = "📊 *СТАТИСТИКА МЕТОДОВ:*\n\n"
    for name, stat in stats.items():
        name_rus = {
            'your_method': 'Ваш метод',
            'logreg_method': 'Лог.регрессия',
            'depth_method': 'Глубинный анализ',
            'markov_method': 'Марковская цепь',
            'lstm_method': 'LSTM'
        }[name]
        if stat['total'] > 0:
            acc = stat['score'] * 100
            msg += f"{name_rus}: {acc:.0f}% ({stat['correct']}/{stat['total']})\n"
        else:
            msg += f"{name_rus}: нет данных\n"
    await update.message.reply_text(msg, parse_mode="Markdown")

async def upload(update: Update, context):
    await update.message.reply_text("📁 Отправьте файл lottery.csv (как документ)")

async def handle_document(update: Update, context):
    doc = update.message.document
    if doc.file_name == 'lottery.csv':
        await update.message.reply_text("🔄 Загружаю...")
        file = await doc.get_file()
        await file.download_to_drive(DATA_FILE)
        await update.message.reply_text(f"✅ Загружен в {DATA_FILE}\nТеперь проверьте /history")
    else:
        await update.message.reply_text(f"❌ Ожидался lottery.csv, получен {doc.file_name}")

async def handle_message(update: Update, context):
    if context.user_data.get('waiting'):
        try:
            nums = [int(x) for x in update.message.text.split()]
            if len(nums) == 6 and all(1 <= x <= 6 for x in nums):
                await update.message.reply_text("🔄 Добавляю...")
                new_num = add_draw_to_file(nums)
                await update.message.reply_text(f"✅ Тираж {new_num} добавлен!\n{nums} | сумма {sum(nums)}")
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
    app.add_handler(CommandHandler("del", delete_last))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("upload", upload))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Бот запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()
