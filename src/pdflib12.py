"""
pdflib12.py  —  ECG plot and PDF export utilities
==================================================
Provides helper functions for visualising 12-lead ECG signals
and exporting them as PNG / PDF files.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import os
from math import ceil

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io


def _ax_plot(ax, x, y, color='k', secs=10, lwidth=1.5, amplitude_ecg=1.8,
             time_ticks=0.2, sample_rate=500, m=["s", "o"], ms=[75, 25]):
    """Render a single ECG lead onto a matplotlib Axes object."""
    ax.plot(x, y, color, linewidth=lwidth)
    ax.set_xticks(np.arange(0, secs + 1, time_ticks))
    ax.xaxis.set_major_locator(MaxNLocator(60))
    ax.yaxis.set_major_locator(MaxNLocator(9))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
    ax.set_xlim(0, secs)


lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def plot_12(ecgTrue, ecgEst, sample_rate=500, title='ECG 12', fontsize=8,
            lead_index=lead_index, lead_order=None, columns=1, speed=50,
            voltage=10, line_width=0.6, m=["s", "o"], ms=[75, 25]):
    """
    Plot true and estimated 12-lead ECG signals side by side.

    Parameters
    ----------
    ecgTrue, ecgEst : np.ndarray  shape (1, T, 12)
    sample_rate     : int    sampling frequency in Hz
    title           : str    figure title
    """
    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(
        ceil(len(lead_index) / columns), columns,
        figsize=((speed / 25) * 10 * columns,
                 (4.1 * voltage / 25) * len(lead_index) / columns),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(hspace=0, wspace=0.04,
                        left=0.14, right=0.98, bottom=0.06, top=0.95)
    fig.suptitle(title)

    if not lead_order:
        lead_order = lead_index

    for i, ax in enumerate(axs.flat):
        t_lead = lead_order[i]
        x1 = np.arange(0, len(ecgTrue[0, :, i]) / sample_rate, 1 / sample_rate)
        x2 = np.arange(0, len(ecgEst[0,  :, i]) / sample_rate, 1 / sample_rate)
        _ax_plot(ax, x1, ecgTrue[0, :, i], color='b', secs=10,
                 lwidth=line_width, amplitude_ecg=1.8, time_ticks=0.2,
                 m=m, ms=ms)
        _ax_plot(ax, x2, ecgEst[0,  :, i], color='g', secs=10,
                 lwidth=line_width, amplitude_ecg=1.8, time_ticks=0.2,
                 m=m, ms=ms)
        ax.set_ylabel(t_lead)
        if i == 0:
            ax.legend(['True', 'Estimated'])


def save_as_png(file_name, path="", dpi=100, layout='tight'):
    """
    Save the current matplotlib figure as a PNG file.

    Parameters
    ----------
    file_name : str   filename without extension
    path      : str   destination directory (default: current directory)
    dpi       : int   dots per inch
    layout    : str   'tight' to include axis labels in the saved image
    """
    plt.ioff()
    plt.savefig(path + file_name + '.png', dpi=dpi, bbox_inches=layout)
    plt.close()


def add_image_to_new_pdf(image_path, output_pdf_path):
    """
    Embed an image file into a new single-page PDF document.

    Parameters
    ----------
    image_path      : str   path to the source image
    output_pdf_path : str   path where the output PDF will be saved
    """
    with Image.open(image_path) as img:
        width, height = img.size

    custom_page_size = (width, height)
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=custom_page_size)
    c.drawImage(image_path, 0, 0, width=width, height=height, mask='auto')
    c.save()

    packet.seek(0)
    new_pdf_reader = PdfReader(packet)
    writer = PdfWriter()
    writer.add_page(new_pdf_reader.pages[0])

    with open(output_pdf_path, 'wb') as f_out:
        writer.write(f_out)

    print(f"PDF saved: '{output_pdf_path}'")
