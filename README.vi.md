**[🇺🇸 English](README.md)**

# Local OpenAI LLM cho Home Assistant

Repository này là một bản fork đặc biệt từ dự án gốc [Local OpenAI LLM](https://github.com/skye-harris/hass_local_openai_llm) của [@skye-harris](https://github.com/skye-harris). Nó duy trì các tính năng cốt lõi đồng thời thêm các cải tiến mạnh mẽ được thiết kế riêng để chạy các mô hình Google Gemini hoàn toàn miễn phí thông qua [Gemini-FastAPI](https://github.com/luuquangvu/ha-addons).

---

## Tính năng nổi bật

- **Mở rộng khả năng nhận biết ngữ cảnh**: Được tối ưu hóa để tận dụng tối đa khả năng xử lý ngữ cảnh dài của mô hình và tối đa hóa hiệu quả lưu bộ nhớ đệm ngữ cảnh (context caching), đảm bảo agent duy trì trí nhớ nhất quán trong suốt các cuộc hội thoại kéo dài.
- **Làm chủ đa phương thức (Multimodal Mastery)**: Gửi văn bản, hình ảnh, âm thanh, video và tệp PDF trực tiếp đến Google Gemini để phân tích và suy luận nâng cao.
- **Hoàn toàn miễn phí & Không cần API Key**: Truy cập các mô hình Google Gemini mạnh mẽ miễn phí như một giải pháp thay thế OpenAI. Không yêu cầu dự án Google Cloud hay API key chính thức (được hỗ trợ bởi [Gemini-FastAPI](https://github.com/luuquangvu/ha-addons)).
- **Tích hợp Home Assistant gốc**: Tích hợp sâu với **Assist**, hỗ trợ gọi công cụ (tool calling/intent handling), đầu vào hình ảnh cho các tác vụ AI và tùy chỉnh nhiệt độ (temperature).
- **Kiểm soát Prompt thủ công**: Toàn quyền kiểm soát các hướng dẫn hệ thống (system instructions) với hỗ trợ Jinja2 template để định hình phản hồi và tính cách của AI một cách chính xác.
- **Tạo hình ảnh**: Tích hợp hỗ trợ tạo hình ảnh trực tiếp thông qua conversation agent hoặc các dịch vụ chuyên dụng.
- **Đầu ra sẵn sàng cho thông báo**: Tích hợp sẵn tính năng loại bỏ emoji, dọn dẹp ký tự nhấn mạnh Markdown và loại bỏ LaTeX để đảm bảo các thông báo qua TTS (chuyển văn bản thành giọng nói) rõ ràng và chất lượng cao.

---

## Điều kiện tiên quyết

- **Home Assistant** đã cài đặt [HACS](https://hacs.xyz/).
- **Gemini-FastAPI Server**: [Tải tại đây](https://github.com/luuquangvu/ha-addons) (khuyên dùng để truy cập Gemini mượt mà nhất).
- **Thông tin đăng nhập Gemini**: Các cookie `__Secure-1PSID` và `__Secure-1PSIDTS` hợp lệ (yêu cầu bởi instance Gemini-FastAPI của bạn).

---

## Cài đặt

### Lựa chọn 1: HACS (Khuyên dùng)

[![Thêm Local OpenAI LLM vào HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=luuquangvu&repository=hass_local_openai_llm&category=integration)

1. Mở **HACS** và chọn **Integrations**.
2. Tìm kiếm **Local OpenAI LLM**.
3. Nếu không tìm thấy, thêm `https://github.com/luuquangvu/hass_local_openai_llm` làm **Custom Repository** (Loại: Integration).
4. Nhấn **Download**, sau đó khởi động lại Home Assistant.

### Lựa chọn 2: Cài đặt thủ công

1. Tải repository này dưới dạng tệp ZIP hoặc clone bằng Git.
2. Sao chép thư mục `custom_components/local_openai` vào thư mục `custom_components/` của Home Assistant.
3. Khởi động lại Home Assistant.

---

## Cấu hình

1. Đi tới **Settings** → **Devices & Services**.
2. Nhấn **Add Integration** và tìm kiếm **Local OpenAI LLM**.
3. Cung cấp thông tin chi tiết về máy chủ của bạn:
   - **Server Name**: Tên thân thiện (ví dụ: `Gemini AI`).
   - **Server URL**: Endpoint đầy đủ (ví dụ: `http://127.0.0.1:8000/v1`). **Lưu ý:** Phải bao gồm hậu tố `/v1`.
   - **API Key**: Không bắt buộc (sử dụng API key nếu đã cấu hình trong tệp `config.yaml` của Gemini-FastAPI).
4. Làm theo trình hướng dẫn để tạo **Conversation Agents** hoặc **AI Tasks**.

---

## Lời cảm ơn

- Dựa trên dự án tuyệt vời [Local OpenAI LLM](https://github.com/skye-harris/hass_local_openai_llm) của [@skye-harris](https://github.com/skye-harris).
- Được hỗ trợ bởi [Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) & [Gemini-FastAPI](https://github.com/luuquangvu/ha-addons).
