#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    """
    Читает лог-файл и извлекает из него:
      - epoch (целое число)
      - train_loss
      - train_acc
      - valid_acc
      - quality
    Возвращает dict со списками по каждой метрике, где индексы соответствуют эпохам.
    """
    re_epoch   = re.compile(r'epoch(\d+)')  # захват epochN
    re_train   = re.compile(r'Train: \[\s*(\d+)/(\d+)\].*Loss\s+([\d\.]+).*Acc@\(.+\)\s+([\d\.]+)%')
    re_train_p = re.compile(r'Train: \[\s*(\d+)/(\d+)\].*Final Prec@1\s+([\d\.]+)%')
    re_valid_p = re.compile(r'Valid: \[\s*(\d+)/(\d+)\].*Final Prec@1\s+([\d\.]+)%')
    re_quality = re.compile(r'Quality\*:\s+([\d\.]+)')

    # Списки для хранения результатов
    epochs        = []
    train_loss    = []
    train_acc     = []
    valid_acc     = []
    quality       = []

    # При чтении лога удобно хранить "текущую" эпоху, когда встретим строку epochN
    current_epoch = None

    # Чтобы аккуратно записывать метрики по эпохам, часто применяют словарь по epoch
    # Однако если идти по порядку, можно просто добавлять в конец списков
    # и индекс i будет соответствовать эпохе.
    # Но т.к. лог может идти последовательно, а строки для одной эпохи могут быть перемешаны,
    # используем промежуточное хранение в словаре, а потом перенесём в финальные списки:
    data_per_epoch = {}  # ключ = номер эпохи, значение = dict с полями train_loss, train_acc, valid_acc, quality
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1) Проверяем, не встретили ли мы строку вида "epoch0"
            m_epoch = re_epoch.search(line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))
                # Инициализируем словарь для этой эпохи
                if current_epoch not in data_per_epoch:
                    data_per_epoch[current_epoch] = {
                        'train_loss': None,
                        'train_acc':  None,
                        'valid_acc':  None,
                        'quality':    None
                    }

            # 2) Парсим train-loss и train-acc
            m_tr = re_train.search(line)
            if m_tr and current_epoch is not None:
                # m_tr.group(1) -- какой-то счётчик батчей (необязательно нужен)
                # m_tr.group(3) -- loss, m_tr.group(4) -- acc
                loss_val = float(m_tr.group(3))
                acc_val  = float(m_tr.group(4))
                data_per_epoch[current_epoch]['train_loss'] = loss_val
                data_per_epoch[current_epoch]['train_acc']  = acc_val

            # 3) Парсим финальную train prec строку
            m_trp = re_train_p.search(line)
            if m_trp and current_epoch is not None:
                # Это финальная точность на обучении
                final_train_prec = float(m_trp.group(3))
                # Если вам нужно именно это значение, можно сохранить, как train_acc
                # Или завести отдельное поле, чтобы сравнивать "on the fly" vs "final"
                data_per_epoch[current_epoch]['train_acc'] = final_train_prec

            # 4) Парсим строку validation
            m_valp = re_valid_p.search(line)
            if m_valp and current_epoch is not None:
                final_val_prec = float(m_valp.group(3))
                data_per_epoch[current_epoch]['valid_acc'] = final_val_prec

            # 5) Парсим quality
            m_q = re_quality.search(line)
            if m_q and current_epoch is not None:
                q_val = float(m_q.group(1))
                data_per_epoch[current_epoch]['quality'] = q_val

    # Теперь переносим данные из data_per_epoch в упорядоченные списки
    # Поскольку эпохи шли по порядку 0,1,2,... то можно отсортировать по ключам
    sorted_epochs = sorted(data_per_epoch.keys())
    for ep in sorted_epochs:
        epochs.append(ep)
        train_loss.append(data_per_epoch[ep]['train_loss'])
        train_acc.append(data_per_epoch[ep]['train_acc'])
        valid_acc.append(data_per_epoch[ep]['valid_acc'])
        quality.append(data_per_epoch[ep]['quality'])

    results = {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'quality': quality
    }
    return results


def plot_learning_curves(results, out_prefix='learning_curves'):
    """
    Рисует несколько графиков на основе результатов парсинга:
      - Epoch vs Train Loss
      - Epoch vs Train Accuracy
      - Epoch vs Valid Accuracy
      - Epoch vs Quality
    Сохраняет их в PNG. Если хотите отобразить, раскомментируйте plt.show().
    """
    epochs = results['epochs']

    # 1) Train loss
    plt.figure()
    plt.plot(epochs, results['train_loss'], marker='o', label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{out_prefix}_train_loss.png')
    #plt.show()

    # 2) Train accuracy
    plt.figure()
    plt.plot(epochs, results['train_acc'], marker='o', color='green', label='Train Acc')
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{out_prefix}_train_acc.png')
    #plt.show()

    # 3) Valid accuracy
    plt.figure()
    plt.plot(epochs, results['valid_acc'], marker='o', color='orange', label='Valid Acc')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{out_prefix}_valid_acc.png')
    #plt.show()

    # 4) Quality
    plt.figure()
    plt.plot(epochs, results['quality'], marker='o', color='red', label='Quality')
    plt.title('Quality')
    plt.xlabel('Epoch')
    plt.ylabel('Quality')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{out_prefix}_quality.png')
    #plt.show()


def main():
    log_file = 'searchs\\kappa_50_3_ops\\kappa_50_3_ops_train.log'  # <-- замените на путь к вашему логу
    results = parse_log_file(log_file)
    plot_learning_curves(results, out_prefix='learning_curves')

if __name__ == '__main__':
    main()
