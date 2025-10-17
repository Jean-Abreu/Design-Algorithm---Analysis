#include <iostream>
#include <vector>
#include <cstdlib>

void insertionSort(std::vector<int>& arr) {
    int n = arr.size();
    for(int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while(j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

int main() {
    //Insertion Sort
    int size, bound;
    std::cout << "Enter the size of the array: ";
    std::cin >> size;
    std::cout << "Enter the upper bound for random numbers: ";
    std::cin >> bound;

    std::vector<int> arr(size);
    for(int i = 0; i < size; i++) {
        arr[i] = rand() % (bound + 1);
    }

    std::cout << "Unsorted array: ";
    for(int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    insertionSort(arr);

    std::cout << "Sorted array: ";
    for(int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
}