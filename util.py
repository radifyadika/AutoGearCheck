import base64

import plotly.graph_objects as go
from PIL import Image
import streamlit as st


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)



def visualize_with_mask(image_np, vis_mask_rgb, binary_mask_np):
    """
    Visualizes original image, detected mask overlay, and binary mask using Plotly toggle buttons.

    Args:
        image_np: np.array original image (RGB)
        vis_mask_rgb: np.array visualized mask overlay (RGB)
        binary_mask_np: np.array binary mask (0/255)
    """
    orig_pil = Image.fromarray(image_np)
    mask_pil = Image.fromarray(vis_mask_rgb)
    binary_mask_pil = Image.fromarray(binary_mask_np)

    width, height = orig_pil.size

    fig = go.Figure()

    # Default image (original)
    fig.add_layout_image(
        dict(
            source=orig_pil,
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            xref="x",
            yref="y",
            sizing="stretch",
            layer="below"
        )
    )

    # Remove all axes and grid
    fig.update_xaxes(
        visible=False,
        range=[0, width],
        constrain="domain"
    )
    fig.update_yaxes(
        visible=False,
        range=[0, height],
        scaleanchor="x",
        scaleratio=1,
        constrain="domain"
    )

    fig.update_layout(
        height=750,
        updatemenus=[
            dict(
                direction='left',
                pad=dict(r=10, t=10),
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
                type="buttons",
                buttons=[
                    dict(label="Original",
                         method="relayout",
                         args=[{"images[0].source": orig_pil}]),
                    dict(label="Deteksi",
                         method="relayout",
                         args=[{"images[0].source": mask_pil}]),
                    dict(label="Mask",
                         method="relayout",
                         args=[{"images[0].source": binary_mask_pil}]),
                ],
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

