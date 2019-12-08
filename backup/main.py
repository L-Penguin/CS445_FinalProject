import torch

def main():
    cnn = MRFCNN()
    for each res:
        cnn.update_content_and_style_img(content, style);

        tv, contents, styles = cnn.forward(syth)
        loss = cnn.calculate_total_loss(tv, contents, styles)
        loss.backward()
        opt.step()
