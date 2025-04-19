from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def calculate_metrics(tp_yes, fn_yes, fp_yes, tn_yes, tp_no, fn_no, fp_no, tn_no, selected_metrics):
    metrics = {}
    total = (tp_yes + fn_yes + fp_yes + tn_yes + tp_no + fn_no + fp_no + tn_no)

    # Overall metrics
    if 'accuracy' in selected_metrics:
        metrics['accuracy'] = (tp_yes + tn_yes + tp_no + tn_no) / total if total > 0 else 0

    # Yes class metrics
    metrics['yes'] = {}
    if 'tpr_yes' in selected_metrics:
        metrics['yes']['tpr'] = tp_yes / (tp_yes + fn_yes) if (tp_yes + fn_yes) > 0 else 0
    if 'tnr_yes' in selected_metrics:
        metrics['yes']['tnr'] = tn_yes / (tn_yes + fp_yes) if (tn_yes + fp_yes) > 0 else 0
    if 'fpr_yes' in selected_metrics:
        metrics['yes']['fpr'] = fp_yes / (fp_yes + tn_yes) if (fp_yes + tn_yes) > 0 else 0
    if 'fnr_yes' in selected_metrics:
        metrics['yes']['fnr'] = fn_yes / (fn_yes + tp_yes) if (fn_yes + tp_yes) > 0 else 0
    if 'ppv_yes' in selected_metrics or 'f1_yes' in selected_metrics:
        metrics['yes']['ppv'] = tp_yes / (tp_yes + fp_yes) if (tp_yes + fp_yes) > 0 else 0
    if 'f1_yes' in selected_metrics:
        tpr = metrics['yes'].get('tpr', tp_yes / (tp_yes + fn_yes) if (tp_yes + fn_yes) > 0 else 0)
        ppv = metrics['yes'].get('ppv', tp_yes / (tp_yes + fp_yes) if (tp_yes + fp_yes) > 0 else 0)
        metrics['yes']['f1'] = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0

    # No class metrics
    metrics['no'] = {}
    if 'tpr_no' in selected_metrics:
        metrics['no']['tpr'] = tp_no / (tp_no + fn_no) if (tp_no + fn_no) > 0 else 0
    if 'tnr_no' in selected_metrics:
        metrics['no']['tnr'] = tn_no / (tn_no + fp_no) if (tn_no + fp_no) > 0 else 0
    if 'fpr_no' in selected_metrics:
        metrics['no']['fpr'] = fp_no / (fp_no + tn_no) if (fp_no + tn_no) > 0 else 0
    if 'fnr_no' in selected_metrics:
        metrics['no']['fnr'] = fn_no / (fn_no + tp_no) if (fn_no + tp_no) > 0 else 0
    if 'ppv_no' in selected_metrics or 'f1_no' in selected_metrics:
        metrics['no']['ppv'] = tp_no / (tp_no + fp_no) if (tp_no + fp_no) > 0 else 0
    if 'f1_no' in selected_metrics:
        tpr = metrics['no'].get('tpr', tp_no / (tp_no + fn_no) if (tp_no + fn_no) > 0 else 0)
        ppv = metrics['no'].get('ppv', tp_no / (tp_no + fp_no) if (tp_no + fp_no) > 0 else 0)
        metrics['no']['f1'] = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0

    app.logger.debug(f"Calculated Metrics: {metrics}")
    return metrics

def plot_metrics(data, metric, plot_type, class_name=None, datasets=None):
    plt.figure(figsize=(8, 5))
    title = f'{metric.replace("_", " ").title()} Comparison'
    if class_name:
        metric_key = metric
        title = f'{metric_key.upper()} ({class_name.capitalize()}) Comparison'
    else:
        metric_key = metric

    if datasets and len(datasets) > 1 and plot_type == 'line':
        dataset_names = datasets
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
        for idx, model_data in enumerate(data):
            model_name = model_data['name']
            values = []
            for ds in model_data['datasets']:
                try:
                    if class_name:
                        values.append(ds['metrics'][class_name][metric_key])
                    else:
                        values.append(ds['metrics'][metric_key])
                except KeyError:
                    values.append(0)
            plt.plot(dataset_names, values, marker='o', label=model_name, color=colors[idx % len(colors)])

        plt.xlabel('Datasets', fontsize=12)
        plt.legend()
    else:
        model_names = [model['name'] for model in data]
        values = []
        for model in data:
            try:
                if class_name:
                    values.append(model['datasets'][0]['metrics'][class_name][metric_key])
                else:
                    values.append(model['datasets'][0]['metrics'][metric_key])
            except KeyError:
                values.append(0)
        plt.bar(model_names, values, color='#3b82f6', edgecolor='black')

    plt.title(title, fontsize=14, pad=15)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_models = int(request.form.get('num_models', 1))
        num_datasets = int(request.form.get('num_datasets', 1))
        selected_metrics = request.form.getlist('metrics')
        plot_type = request.form.get('plot_type', 'bar')
        datasets = [request.form.get(f'dataset_name_{i}', f'Dataset {i+1}') for i in range(num_datasets)]

        app.logger.debug(f"Selected Metrics: {selected_metrics}")

        if not selected_metrics:
            return render_template('index.html', error="Please select at least one metric")

        models_data = []
        for i in range(num_models):
            model_name = request.form.get(f'model_name_{i}', f'Model {i+1}')
            custom_name = request.form.get(f'custom_model_name_{i}', '')
            name = custom_name if custom_name else model_name
            model_data = {'name': name, 'datasets': []}

            for j in range(num_datasets):
                try:
                    tp_yes = int(request.form.get(f'tp_yes_{i}_{j}', 0))
                    fn_yes = int(request.form.get(f'fn_yes_{i}_{j}', 0))
                    fp_yes = int(request.form.get(f'fp_yes_{i}_{j}', 0))
                    tn_yes = int(request.form.get(f'tn_yes_{i}_{j}', 0))
                    tp_no = int(request.form.get(f'tp_no_{i}_{j}', 0))
                    fn_no = int(request.form.get(f'fn_no_{i}_{j}', 0))
                    fp_no = int(request.form.get(f'fp_no_{i}_{j}', 0))
                    tn_no = int(request.form.get(f'tn_no_{i}_{j}', 0))

                    metrics = calculate_metrics(tp_yes, fn_yes, fp_yes, tn_yes, tp_no, fn_no, fp_no, tn_no, selected_metrics)
                    model_data['datasets'].append({
                        'name': datasets[j],
                        'yes': {'tp': tp_yes, 'fn': fn_yes, 'fp': fp_yes, 'tn': tn_yes},
                        'no': {'tp': tp_no, 'fn': fn_no, 'fp': fp_no, 'tn': tn_no},
                        'metrics': metrics
                    })
                except ValueError:
                    return render_template('index.html', error="Please enter valid numbers for confusion matrix values")

            models_data.append(model_data)

        app.logger.debug(f"Models Data: {models_data}")

        plots = {}
        for metric in selected_metrics:
            class_name = None
            if metric.startswith(('tpr_', 'tnr_', 'fpr_', 'fnr_', 'ppv_', 'f1_')):
                class_name = metric.split('_')[-1]
                metric_key = metric.split('_')[0]
            else:
                metric_key = metric
            try:
                plots[metric] = plot_metrics(models_data, metric_key, plot_type, class_name, datasets if metric == 'accuracy' else None)
            except Exception as e:
                return render_template('index.html', error=f"Error generating plot for {metric}: {str(e)}")

        return render_template('results.html', models_data=models_data, plots=plots, selected_metrics=selected_metrics, datasets=datasets)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)