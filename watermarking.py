import numpy as np
import pywt
import cv2
from scipy.fftpack import dct, idct
from hashlib import md5
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def hamming_encode(data):
    """
    Mã hóa văn bản watermark bằng mã Hamming.
    """
    binary_data = ''.join(format(ord(c), '08b') for c in data)  # Chuyển văn bản sang nhị phân
    encoded = []
    for bit in binary_data:
        encoded.append(bit)
        encoded.append(bit)  # Nhân đôi để mã hóa đơn giản
    return ''.join(encoded)

def chaotic_map_encrypt(image, key):
    """
    Mã hóa ảnh bằng bản đồ hỗn loạn 2D.
    """
    np.random.seed(hash(key) % 2**32)
    height, width = image.shape[:2]
    x_seq = np.random.permutation(height)
    y_seq = np.random.permutation(width)
    encrypted_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            encrypted_image[i, j] = image[x_seq[i], y_seq[j]]
    return encrypted_image

def chaotic_map_decrypt(image, key):
    """
    Giải mã ảnh đã mã hóa bằng bản đồ hỗn loạn 2D.
    """
    np.random.seed(hash(key) % 2**32)
    height, width = image.shape[:2]
    x_seq = np.random.permutation(height)
    y_seq = np.random.permutation(width)
    decrypted_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            decrypted_image[x_seq[i], y_seq[j]] = image[i, j]
    return decrypted_image

def rgb_to_yiq(image):
    """
    Chuyển đổi ảnh từ không gian RGB sang YIQ.
    """
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                   [0.596, -0.274, -0.322],
                                   [0.211, -0.523, 0.312]])
    return np.dot(image, conversion_matrix.T)

def yiq_to_rgb(image):
    """
    Chuyển đổi ảnh từ không gian YIQ sang RGB.
    """
    conversion_matrix = np.array([[1, 0.956, 0.621],
                                   [1, -0.272, -0.647],
                                   [1, -1.106, 1.703]])
    return np.dot(image, conversion_matrix.T)

def calculate_metrics(original_logo, extracted_logo):
    """
    Tính toán PSNR và SSIM giữa logo gốc và logo đã trích xuất.
    """
    if original_logo.shape != extracted_logo.shape:
        extracted_logo = cv2.resize(extracted_logo, (original_logo.shape[1], original_logo.shape[0]))
    psnr_value = psnr(original_logo, extracted_logo, data_range=255)
    ssim_value = ssim(original_logo, extracted_logo, data_range=255)
    return psnr_value, ssim_value

def embed_watermark_secure(cover_image, logo_image, text_watermark, oef):
    """
    Nhúng watermark (logo + văn bản) vào thành phần Y của ảnh YIQ.
    """
    # Chuyển RGB → YIQ
    yiq_image = rgb_to_yiq(cover_image)
    y_component = yiq_image[:, :, 0]

    # Hash logo bằng MD5 để tăng cường bảo mật
    md5_hash = md5(logo_image.tobytes()).hexdigest()  # Hash logo dưới dạng chuỗi hex
    hashed_logo = np.array([int(char, 16) for char in md5_hash if char.isalnum()])  # Chuyển hex → số

    # Biến đổi LWT và DCT trên thành phần Y
    coeffs = pywt.wavedec2(y_component, 'haar', level=3)
    cA3, (cH3, cV3, cD3) = coeffs[0], coeffs[1]

    dct_coeffs = dct(dct(cH3.T, norm='ortho').T, norm='ortho')

    # Nhúng logo watermark vào vùng tần số trung bình
    logo_resized = cv2.resize(logo_image, (dct_coeffs.shape[1], dct_coeffs.shape[0]))
    dct_coeffs[10:20, 10:20] += oef * logo_resized[10:20, 10:20]

    # Nhúng văn bản watermark bằng mã Hamming
    encoded_text = hamming_encode(text_watermark)
    for i, bit in enumerate(encoded_text):
        row, col = (i // dct_coeffs.shape[1]) % dct_coeffs.shape[0], i % dct_coeffs.shape[1]
        dct_coeffs[row, col] += oef * (1 if bit == '1' else -1)

    # Khôi phục thành phần Y sau nhúng
    cH3_mod = idct(idct(dct_coeffs.T, norm='ortho').T, norm='ortho')
    coeffs[1] = (cH3_mod, cV3, cD3)
    y_modified = pywt.waverec2(coeffs, 'haar')

    # Ghi đè thành phần Y và chuyển lại sang RGB
    yiq_image[:, :, 0] = np.clip(y_modified, 0, 255)
    watermarked_image = yiq_to_rgb(yiq_image)

    return np.clip(watermarked_image, 0, 255).astype(np.uint8)


def extract_watermark(watermarked_image, cover_image, oef, logo_shape):
    """
    Trích xuất watermark từ ảnh đã nhúng watermark.
    """
    # Chuyển đổi RGB → YIQ và lấy kênh Y
    yiq_watermarked = rgb_to_yiq(watermarked_image)
    y_watermarked = yiq_watermarked[:, :, 0]

    yiq_original = rgb_to_yiq(cover_image)
    y_original = yiq_original[:, :, 0]

    # Biến đổi LWT và DCT
    coeffs_original = pywt.wavedec2(y_original, 'haar', level=3)
    coeffs_watermarked = pywt.wavedec2(y_watermarked, 'haar', level=3)

    cA3_orig, (cH3_orig, _, _) = coeffs_original[0], coeffs_original[1]
    cA3_wm, (cH3_wm, _, _) = coeffs_watermarked[0], coeffs_watermarked[1]

    dct_orig = dct(dct(cH3_orig.T, norm='ortho').T, norm='ortho')
    dct_wm = dct(dct(cH3_wm.T, norm='ortho').T, norm='ortho')

    # Trích xuất watermark
    logo_dct = (dct_wm - dct_orig) / oef
    logo = idct(idct(logo_dct.T, norm='ortho').T, norm='ortho')

    # Resize logo về kích thước ban đầu
    return np.clip(cv2.resize(logo, (logo_shape[1], logo_shape[0])), 0, 255).astype(np.uint8)
def calculate_metrics_between_images(original_image, watermarked_image):
    """
    Tính toán PSNR và SSIM giữa ảnh gốc và ảnh sau khi nhúng watermark.
    """
    # Resize ảnh nếu cần
    min_size = 7
    if original_image.shape[0] < min_size or original_image.shape[1] < min_size:
        original_image = cv2.resize(original_image, (min_size, min_size))
    if watermarked_image.shape[0] < min_size or watermarked_image.shape[1] < min_size:
        watermarked_image = cv2.resize(watermarked_image, (min_size, min_size))

    # Tính PSNR
    psnr_value = psnr(original_image, watermarked_image, data_range=255)

    # Tính SSIM với win_size được đặt cụ thể
    ssim_value = ssim(original_image, watermarked_image, multichannel=True, win_size=3)

    return psnr_value, ssim_value

# Main function
if __name__ == "__main__":
    # Load ảnh gốc và logo
    cover = cv2.imread('cover_image.jpg')
    logo = cv2.imread('logo_image.jpg', cv2.IMREAD_GRAYSCALE)

    # Kiểm tra ảnh có được tải không
    if cover is None or logo is None:
        raise FileNotFoundError("Không tìm thấy ảnh gốc hoặc logo. Hãy kiểm tra đường dẫn.")

    # Tham số
    oef = 0.2
    key = "secure_chaos_key"
    text_watermark = "SecureWatermark"

    # Nhúng watermark
    print("Đang nhúng watermark...")
    watermarked = embed_watermark_secure(cover, logo, text_watermark, oef)
    cv2.imwrite('watermarked_image.jpg', watermarked)
    # Tính toán PSNR và SSIM giữa ảnh gốc và ảnh sau khi nhúng watermark
    print("Đang tính toán chỉ số PSNR và SSIM...")
    psnr_val, ssim_val = calculate_metrics_between_images(cover, watermarked)
    print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    # Mã hóa ảnh gốc
    print("Đang mã hóa ảnh gốc...")
    encrypted_cover = chaotic_map_encrypt(cover, key)
    cv2.imwrite('encrypted_cover.jpg', encrypted_cover)

    # Giải mã ảnh
    print("Đang giải mã ảnh...")
    decrypted_cover = chaotic_map_decrypt(encrypted_cover, key)
    cv2.imwrite('decrypted_cover.jpg', decrypted_cover)

    # Trích xuất logo watermark
    print("Đang trích xuất watermark...")
    extracted_logo = extract_watermark(watermarked, cover, oef, logo.shape)

    # Đảm bảo kích thước logo trích xuất phù hợp
    extracted_logo_resized = cv2.resize(extracted_logo, (logo.shape[1], logo.shape[0]))



    # Lưu logo đã trích xuất
    cv2.imwrite('extracted_logo.jpg', extracted_logo_resized)

    print("Ảnh mã hóa và giải mã được lưu như sau:")
    print("- Ảnh đã mã hóa: encrypted_cover.jpg")
    print("- Ảnh đã giải mã: decrypted_cover.jpg")
    print("- Ảnh watermark: watermarked_image.jpg")
    print("- Logo đã trích xuất: extracted_logo.jpg")
