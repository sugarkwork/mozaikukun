from mozaikukun_ui import full_ui

if __name__=="__main__":
    demo = full_ui()
    demo.launch()

"""
demo = gr.Interface(
    fn=mosaic_process,
    inputs=[
        gr.Image(),
        gr.Radio(choices=['raw', 'mosaic', 'white'], value='mosaic'),
        gr.Radio(choices=['raw', 'mosaic', 'white'], value='mosaic'),
        gr.Radio(choices=['raw', 'mosaic', 'white'], value='mosaic'),
        gr.Radio(choices=['raw', 'mosaic', 'white'], value='raw'),
        gr.Radio(choices=['raw', 'mosaic', 'white'], value='raw'),
    ],
    outputs=["image"])

demo.launch()
"""