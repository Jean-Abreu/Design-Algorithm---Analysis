import random
import time
import pandas as pd
import matplotlib.pyplot as plt

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        merge_sort(left)
        merge_sort(right)

        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

def generate_random_array(size, bound):
    return [random.randint(1, bound) for _ in range(size)]

sizes = [1000, 5000, 10000]
bound = 9
results = []

for size in sizes:
    # Best Case: Already Sorted
    arr_best = list(range(1, size + 1))
    arr_best_merge = arr_best.copy()
    start = time.time()
    insertion_sort(arr_best)
    best_time = (time.time() - start) * 1000

    start = time.time()
    merge_sort(arr_best_merge)
    best_time_merge = (time.time() - start) * 1000

    # Worst Case: Reverse Sorted
    arr_worst = list(range(size, 0, -1))
    arr_worst_merge = arr_worst.copy()
    start = time.time()
    insertion_sort(arr_worst)
    worst_time = (time.time() - start) * 1000

    start = time.time()
    merge_sort(arr_worst_merge)
    worst_time_merge = (time.time() - start) * 1000

    # Average Case: Randomly Ordered
    arr_avg = generate_random_array(size, bound)
    arr_avg_merge = arr_avg.copy()
    start = time.time()
    insertion_sort(arr_avg)
    avg_time = (time.time() - start) * 1000

    start = time.time()
    merge_sort(arr_avg_merge)
    avg_time_merge = (time.time() - start) * 1000

    results.append([size, 'Best', f"{best_time:.3f}", f"{best_time_merge:.3f}"])
    results.append([size, 'Worst', f"{worst_time:.3f}", f"{worst_time_merge:.3f}"])
    results.append([size, 'Average', f"{avg_time:.3f}", f"{avg_time_merge:.3f}"])

df = pd.DataFrame(results, columns=['Input Size', 'Case', 'Insertion Sort Time (ms)', 'Merge Sort Time (ms)'])

# Display table using matplotlib
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))
plt.title("Insertion Sort vs Merge Sort Timing Table")
plt.show()