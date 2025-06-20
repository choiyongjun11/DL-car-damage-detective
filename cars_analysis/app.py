from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from inference import run_inference

app = Flask(__name__)

# 폴더 경로 설정
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# 폴더가 없으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    uploaded = request.files.getlist('cropped_images')
    app.logger.debug(f"Received files: {[f.filename for f in uploaded]}")
    if not uploaded:
        return "❌ 크롭된 이미지가 없습니다. '분석 제출' 전에 반드시 크롭하세요.", 400

    results = []
    for file in uploaded:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(in_path)

        result_name = f"result_{filename}"
        out_path = os.path.join(app.config['RESULT_FOLDER'], result_name)

        try:
            part, damage = run_inference(in_path, out_path)
        except Exception as e:
            app.logger.error(f"Inference error for {filename}: {e}")
            part, damage = "분석 실패", str(e)

        results.append({
            "original": filename,
            "result_img": result_name,
            "part": part,
            "damage": damage
        })

    return render_template('result.html', results=results)

if __name__ == '__main__':
    # 디버그 켜고 0.0.0.0 바인딩
    app.run(host='0.0.0.0', port=5000, debug=True)
