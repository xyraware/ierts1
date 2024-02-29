import numpy as np
import cv2
import random


def expand_image(used_image, expand_factor=2):
    """
    Функция для расширения изображения.

    :param used_image: Исходное изображение.
    :param expand_factor: Фактор расширения.
    :return: Расширенное изображение.
    """
    expanded_image = cv2.resize(used_image, None, fx=expand_factor, fy=expand_factor, interpolation=cv2.INTER_LINEAR)
    return expanded_image


def salt_and_pepper_noise(used_image, amount=0.05):
    """
    Функция для наложения шума типа "соль-перец" на изображение.

    :param used_image: Исходное изображение.
    :param amount: Процент пикселей, которые будут затронуты шумом.
    :return: Изображение с наложенным шумом.
    """
    noisy_image = np.copy(used_image)
    height, width, _ = noisy_image.shape
    num_pixels = int(height * width * amount)
    for _ in range(num_pixels):
        rand_x = random.randint(0, width - 1)
        rand_y = random.randint(0, height - 1)
        rand_value = random.randint(0, 1) * 255
        noisy_image[rand_y, rand_x] = [rand_value, rand_value, rand_value]
    return noisy_image


def gaussian_noise(used_image, mean=0, std=25):
    """
    Функция для наложения шума Гаусса на изображение.

    :param used_image: Исходное изображение.
    :param mean: Среднее значение для распределения Гаусса.
    :param std: Стандартное отклонение для распределения Гаусса.
    :return: Изображение с наложенным шумом Гаусса.
    """
    noisy_image = np.copy(used_image)
    height, width, _ = noisy_image.shape
    gaussian_noise = np.random.normal(mean, std, (height, width, 3))
    noisy_image = np.clip(noisy_image + gaussian_noise, 0, 255).astype(np.uint8)
    return noisy_image


def median_filter(used_image, kernel_size=3):
    """
    Медианный фильтр для удаления шума типа "соль-перец".

    :param used_image: Исходное изображение с шумом.
    :param kernel_size: Размер ядра фильтра.
    :return: Изображение после применения медианного фильтра.
    """
    filtered_image = cv2.medianBlur(used_image, kernel_size)
    return filtered_image


def gaussian_filter(used_image, kernel_size=(5, 5), sigma=1.0):
    """
    Фильтр Гаусса для удаления шума Гаусса.

    :param used_image: Исходное изображение с шумом.
    :param kernel_size: Размер ядра фильтра.
    :param sigma: Стандартное отклонение Гауссовского фильтра.
    :return: Изображение после применения фильтра Гаусса.
    """
    filtered_image = cv2.GaussianBlur(used_image, kernel_size, sigma)
    return filtered_image


def blur_filter(used_image, kernel_size=(5, 5)):
    """
    Функция для применения фильтра размытия к изображению.

    :param used_image: Исходное изображение.
    :param kernel_size: Размер ядра фильтра размытия.
    :return: Размытое изображение.
    """
    blurred_image = cv2.blur(used_image, kernel_size)
    return blurred_image


def sharpen_filter(used_image):
    """
    Функция для применения фильтра резкости к изображению.

    :param used_image: Исходное изображение.
    :return: Резкое изображение.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(used_image, -1, kernel)
    return sharpened_image


image = cv2.imread('1.jpg')
expanded = expand_image(image)

noisy_image_salt_pepper = salt_and_pepper_noise(image)
noisy_image_gaussian = gaussian_noise(image)

filtered_image_salt_pepper = median_filter(noisy_image_salt_pepper)
filtered_image_gaussian = gaussian_filter(noisy_image_gaussian)

blurred_image = blur_filter(image)
sharpened_image = sharpen_filter(image)

cv2.imshow("Expended image", expanded)
cv2.imshow('Original Image', image)
cv2.imshow('Salt and Pepper Noisy Image', noisy_image_salt_pepper)
cv2.imshow('Gaussian Noisy Image', noisy_image_gaussian)
cv2.imshow('Salt and Pepper Filtered Image', filtered_image_salt_pepper)
cv2.imshow('Gaussian Filtered Image', filtered_image_gaussian)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Sharpened Image', sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
