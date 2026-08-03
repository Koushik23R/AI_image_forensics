import React, { Suspense, useEffect, useMemo, useState } from 'react';
import axios from 'axios';

const MonitorLineChart = React.lazy(() => import('./components/MonitorLineChart'));
const PlotlyChart = React.lazy(() => import('./components/PlotlyChart'));

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const api = axios.create({ baseURL: API_BASE_URL });

const navItems = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'dataset', label: 'Dataset Upload' },
  { id: 'train', label: 'Train Model' },
  { id: 'monitor', label: 'Training Monitor' },
  { id: 'prediction', label: 'Prediction' },
  { id: 'statistics', label: 'Statistics' },
];

function formatDate(value) {
  if (!value) return 'N/A';
  return new Date(value * 1000).toLocaleString();
}

function clampText(value, fallback = '—') {
  if (value === null || value === undefined || value === '') return fallback;
  return String(value);
}

function MetricCard({ title, value, helper }) {
  return (
    <div className="metric-card">
      <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{title}</div>
      <div className="mt-3 text-2xl font-semibold text-white">{value}</div>
      {helper ? <div className="mt-2 text-sm text-slate-400">{helper}</div> : null}
    </div>
  );
}

function SectionHeader({ title, subtitle, action }) {
  return (
    <div className="mb-6 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
      <div>
        <h2 className="text-2xl font-semibold text-white">{title}</h2>
        {subtitle ? <p className="mt-2 max-w-3xl text-sm text-slate-400">{subtitle}</p> : null}
      </div>
      {action ? <div>{action}</div> : null}
    </div>
  );
}

function FolderSummary({ title, files }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
      <div className="text-sm font-medium text-white">{title}</div>
      <div className="mt-1 text-xs text-slate-400">{files.length} images selected</div>
      <div className="mt-3 max-h-40 space-y-2 overflow-auto text-xs text-slate-300">
        {files.slice(0, 6).map((file) => (
          <div key={file.webkitRelativePath || file.name} className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
            {file.webkitRelativePath || file.name}
          </div>
        ))}
        {files.length > 6 ? <div className="text-slate-500">+ {files.length - 6} more</div> : null}
      </div>
    </div>
  );
}

function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [datasets, setDatasets] = useState({});
  const [models, setModels] = useState({});
  const [jobs, setJobs] = useState({});
  const [loadingOverview, setLoadingOverview] = useState(true);
  const [overviewError, setOverviewError] = useState('');
  const [lastDatasetId, setLastDatasetId] = useState('');
  const [lastJobId, setLastJobId] = useState('');
  const [selectedMonitorJobId, setSelectedMonitorJobId] = useState('');

  const [realFolderFiles, setRealFolderFiles] = useState([]);
  const [fakeFolderFiles, setFakeFolderFiles] = useState([]);
  const [datasetGeneratorType, setDatasetGeneratorType] = useState('CIFAKE');
  const [datasetImageSize, setDatasetImageSize] = useState('128,128');
  const [uploadingDataset, setUploadingDataset] = useState(false);
  const [datasetMessage, setDatasetMessage] = useState('');

  const [trainDatasetId, setTrainDatasetId] = useState('');
  const [trainEpochs, setTrainEpochs] = useState(4);
  const [trainingMessage, setTrainingMessage] = useState('');
  const [startingTraining, setStartingTraining] = useState(false);

  const [monitorSnapshot, setMonitorSnapshot] = useState(null);
  const [monitorError, setMonitorError] = useState('');

  const [predictionModelId, setPredictionModelId] = useState('');
  const [predictionFile, setPredictionFile] = useState(null);
  const [predictionPreview, setPredictionPreview] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionMessage, setPredictionMessage] = useState('');
  const [runningPrediction, setRunningPrediction] = useState(false);

  const [statisticsModelId, setStatisticsModelId] = useState('');
  const [statisticsResult, setStatisticsResult] = useState(null);
  const [statisticsMessage, setStatisticsMessage] = useState('');
  const [loadingStatistics, setLoadingStatistics] = useState(false);

  useEffect(() => {
    refreshOverview();
  }, []);

  useEffect(() => {
    const datasetKeys = Object.keys(datasets);
    if (!trainDatasetId && datasetKeys.length > 0) {
      setTrainDatasetId(datasetKeys[datasetKeys.length - 1]);
    }
  }, [datasets, trainDatasetId]);

  useEffect(() => {
    const modelKeys = Object.keys(models);
    if (modelKeys.length > 0) {
      const firstModelId = modelKeys[0];
      if (!predictionModelId) setPredictionModelId(firstModelId);
      if (!statisticsModelId) setStatisticsModelId(firstModelId);
    }
  }, [models, predictionModelId, statisticsModelId]);

  useEffect(() => {
    if (!selectedMonitorJobId) return undefined;

    let active = true;
    const loadStatus = async () => {
      try {
        const response = await api.get(`/api/train/status/${selectedMonitorJobId}`);
        if (active) {
          setMonitorSnapshot(response.data);
          setMonitorError('');
        }
      } catch (error) {
        if (active) setMonitorError(error?.response?.data?.detail || error.message);
      }
    };

    loadStatus();
    const interval = window.setInterval(loadStatus, 2000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [selectedMonitorJobId]);

  useEffect(() => {
    if (!predictionFile) {
      setPredictionPreview('');
      return undefined;
    }

    const objectUrl = URL.createObjectURL(predictionFile);
    setPredictionPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [predictionFile]);

  const overviewStats = useMemo(() => {
    const datasetValues = Object.values(datasets);
    const modelValues = Object.values(models);
    const jobValues = Object.values(jobs);

    return {
      datasetCount: datasetValues.length,
      modelCount: modelValues.length,
      jobCount: jobValues.length,
      activeJobs: jobValues.filter((job) => ['pending', 'running'].includes(job.status)).length,
      completedJobs: jobValues.filter((job) => job.status === 'completed').length,
      latestDataset: datasetValues[datasetValues.length - 1] || null,
      latestModel: modelValues[modelValues.length - 1] || null,
      latestJob: jobValues[jobValues.length - 1] || null,
    };
  }, [datasets, models, jobs]);

  async function refreshOverview() {
    setLoadingOverview(true);
    try {
      const [datasetResponse, modelResponse, jobResponse] = await Promise.all([
        api.get('/api/datasets'),
        api.get('/api/models'),
        api.get('/api/jobs'),
      ]);

      setDatasets(datasetResponse.data.datasets || {});
      setModels(modelResponse.data.models || {});
      setJobs(jobResponse.data.jobs || {});

      const datasetKeys = Object.keys(datasetResponse.data.datasets || {});
      const jobKeys = Object.keys(jobResponse.data.jobs || {});
      setLastDatasetId(datasetKeys[datasetKeys.length - 1] || '');
      setLastJobId(jobKeys[jobKeys.length - 1] || '');
      if (!selectedMonitorJobId && jobKeys.length > 0) setSelectedMonitorJobId(jobKeys[jobKeys.length - 1]);
      setOverviewError('');
    } catch (error) {
      setOverviewError(error?.response?.data?.detail || error.message);
    } finally {
      setLoadingOverview(false);
    }
  }

  function handleFolderSelection(event, setFiles) {
    setFiles(Array.from(event.target.files || []));
  }

  async function handleDatasetUpload(event) {
    event.preventDefault();
    if (!realFolderFiles.length || !fakeFolderFiles.length) {
      setDatasetMessage('Select both a real folder and a fake folder before uploading.');
      return;
    }

    const formData = new FormData();
    realFolderFiles.forEach((file) => formData.append('real_files', file, file.name));
    fakeFolderFiles.forEach((file) => formData.append('fake_files', file, file.name));
    formData.append('generator_type', datasetGeneratorType);
    formData.append('img_size', datasetImageSize);

    setUploadingDataset(true);
    setDatasetMessage('Uploading dataset...');
    try {
      const response = await api.post('/api/dataset/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const created = response.data;
      setDatasetMessage(`Dataset ${created.dataset_id} uploaded and processed.`);
      await refreshOverview();
      setTrainDatasetId(created.dataset_id);
      setActivePage('train');
    } catch (error) {
      setDatasetMessage(error?.response?.data?.detail || error.message);
    } finally {
      setUploadingDataset(false);
    }
  }

  async function handleStartTraining(event) {
    event.preventDefault();
    if (!trainDatasetId) {
      setTrainingMessage('Select a dataset id before starting training.');
      return;
    }

    setStartingTraining(true);
    setTrainingMessage('Starting training job...');
    try {
      const response = await api.post('/api/train', {
        dataset_id: trainDatasetId,
        hyperparameters: { epochs: Number(trainEpochs) },
      });
      const jobId = response.data.job_id;
      setTrainingMessage(`Training job ${jobId} started.`);
      setSelectedMonitorJobId(jobId);
      setLastJobId(jobId);
      await refreshOverview();
      setActivePage('monitor');
    } catch (error) {
      setTrainingMessage(error?.response?.data?.detail || error.message);
    } finally {
      setStartingTraining(false);
    }
  }

  async function refreshMonitorNow() {
    if (!selectedMonitorJobId) return;
    try {
      const response = await api.get(`/api/train/status/${selectedMonitorJobId}`);
      setMonitorSnapshot(response.data);
      setMonitorError('');
    } catch (error) {
      setMonitorError(error?.response?.data?.detail || error.message);
    }
  }

  async function handlePrediction(event) {
    event.preventDefault();
    if (!predictionModelId) {
      setPredictionMessage('Select a model before predicting.');
      return;
    }
    if (!predictionFile) {
      setPredictionMessage('Choose an image to predict.');
      return;
    }

    const formData = new FormData();
    formData.append('model_id', predictionModelId);
    formData.append('file', predictionFile);

    setRunningPrediction(true);
    setPredictionMessage('Running prediction...');
    try {
      const response = await api.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPredictionResult(response.data);
      setPredictionMessage('Prediction completed successfully.');
    } catch (error) {
      setPredictionMessage(error?.response?.data?.detail || error.message);
    } finally {
      setRunningPrediction(false);
    }
  }

  async function handleStatisticsLoad() {
    if (!statisticsModelId) {
      setStatisticsMessage('Select a model before loading statistics.');
      return;
    }

    setLoadingStatistics(true);
    setStatisticsMessage('Loading statistical analysis...');
    try {
      const response = await api.get(`/api/statistics/${statisticsModelId}`);
      setStatisticsResult(response.data);
      setStatisticsMessage('Statistics loaded successfully.');
    } catch (error) {
      setStatisticsMessage(error?.response?.data?.detail || error.message);
    } finally {
      setLoadingStatistics(false);
    }
  }

  const monitorHistory = monitorSnapshot?.metrics_history || [];
  const monitorChartData = {
    labels: monitorHistory.map((entry, index) => `Epoch ${entry.epoch || index + 1}`),
    datasets: [
      { label: 'Loss', data: monitorHistory.map((entry) => entry.loss), borderColor: '#22d3ee', backgroundColor: 'rgba(34, 211, 238, 0.12)', tension: 0.35, fill: true },
      { label: 'Validation Loss', data: monitorHistory.map((entry) => entry.val_loss), borderColor: '#f472b6', backgroundColor: 'rgba(244, 114, 182, 0.10)', tension: 0.35, fill: true },
      { label: 'Accuracy', data: monitorHistory.map((entry) => entry.accuracy), borderColor: '#34d399', backgroundColor: 'rgba(52, 211, 153, 0.10)', tension: 0.35, fill: true },
      { label: 'Validation Accuracy', data: monitorHistory.map((entry) => entry.val_accuracy), borderColor: '#fbbf24', backgroundColor: 'rgba(251, 191, 36, 0.10)', tension: 0.35, fill: true },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#cbd5e1' } } },
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.08)' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.08)' } },
    },
  };

  const predictionFFTPlot = predictionResult?.fft_visualization
    ? [{ z: predictionResult.fft_visualization, type: 'heatmap', colorscale: 'Cividis', showscale: true, hoverinfo: 'skip' }]
    : [];

  const statisticsROCPlot = statisticsResult?.roc_curve
    ? [{ x: statisticsResult.roc_curve.fpr, y: statisticsResult.roc_curve.tpr, type: 'scatter', mode: 'lines', line: { color: '#22d3ee', width: 3 }, fill: 'tozeroy', name: 'ROC' }]
    : [];

  const statisticsBoxPlot = statisticsResult?.sample_distributions
    ? [
        { y: statisticsResult.sample_distributions.real_features, type: 'box', name: 'Real', marker: { color: '#34d399' } },
        { y: statisticsResult.sample_distributions.fake_features, type: 'box', name: 'Fake', marker: { color: '#f472b6' } },
      ]
    : [];

  const activeDatasetEntries = Object.entries(datasets);
  const activeModelEntries = Object.entries(models);
  const activeJobEntries = Object.entries(jobs);

  return (
    <div className="min-h-screen text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-[1600px] flex-col lg:flex-row">
        <aside className="border-b border-white/10 bg-slate-950/80 px-5 py-6 backdrop-blur-xl lg:min-h-screen lg:w-80 lg:border-b-0 lg:border-r">
          <div className="glass-panel p-5">
            <div className="text-xs uppercase tracking-[0.3em] text-cyan-300">AI Image Forensics</div>
            <h1 className="mt-3 text-2xl font-semibold text-white">Localhost Control Center</h1>
            <p className="mt-3 text-sm leading-6 text-slate-400">Dual-branch CNN pipeline, FFT preprocessing, background training, and statistical reporting.</p>
          </div>

          <nav className="mt-6 space-y-2">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`flex w-full items-center justify-between rounded-2xl px-4 py-3 text-left transition ${
                  activePage === item.id ? 'bg-cyan-400 text-slate-950 shadow-glow' : 'bg-white/5 text-slate-300 hover:bg-white/10 hover:text-white'
                }`}
              >
                <span className="font-medium">{item.label}</span>
                <span className="text-xs uppercase tracking-[0.2em] opacity-70">{item.id}</span>
              </button>
            ))}
          </nav>

          <div className="mt-6 grid gap-3">
            <div className="glass-panel p-4">
              <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Backend</div>
              <div className="mt-2 text-sm text-white">{API_BASE_URL}</div>
            </div>
            <button className="button-secondary" onClick={refreshOverview}>Refresh Workspace Data</button>
          </div>
        </aside>

        <main className="flex-1 px-4 py-6 sm:px-6 lg:px-8">
          <div className="glass-panel px-6 py-5">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <div className="text-xs uppercase tracking-[0.3em] text-cyan-300">AI Generated Image Forensics</div>
                <h2 className="mt-2 text-3xl font-semibold text-white">{navItems.find((item) => item.id === activePage)?.label}</h2>
                <p className="mt-2 max-w-4xl text-sm text-slate-400">FastAPI on port 8000 with a React + Vite dashboard on port 5173. Upload folders of real and fake images instead of tagging each file manually.</p>
              </div>
              <div className="flex flex-wrap gap-3">
                <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Last Dataset</div>
                  <div className="mt-1 text-sm text-white">{lastDatasetId ? lastDatasetId.slice(0, 8) : 'None'}</div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Last Job</div>
                  <div className="mt-1 text-sm text-white">{lastJobId ? lastJobId.slice(0, 8) : 'None'}</div>
                </div>
              </div>
            </div>
          </div>

          {overviewError ? <div className="mt-5 rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">{overviewError}</div> : null}
          {loadingOverview ? <div className="mt-5 text-sm text-slate-400">Loading backend state...</div> : null}

          {activePage === 'dashboard' ? (
            <div className="mt-6 space-y-6">
              <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
                <MetricCard title="Datasets" value={overviewStats.datasetCount} helper="Registered in backend memory" />
                <MetricCard title="Models" value={overviewStats.modelCount} helper="Saved model metadata" />
                <MetricCard title="Training Jobs" value={overviewStats.jobCount} helper="All background jobs" />
                <MetricCard title="Active Jobs" value={overviewStats.activeJobs} helper="Pending or running" />
                <MetricCard title="Completed Jobs" value={overviewStats.completedJobs} helper="Finished jobs" />
              </section>

              <section className="grid gap-6 xl:grid-cols-3">
                <div className="glass-panel p-6 xl:col-span-2">
                  <SectionHeader title="Project Overview" subtitle="The notebook has been converted into a service-backed application with local uploads, in-memory metadata, and asynchronous training updates." />
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                      <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Latest Dataset</div>
                      <div className="mt-2 text-sm text-white">{overviewStats.latestDataset?.name || 'None yet'}</div>
                      <div className="mt-1 text-xs text-slate-500">{overviewStats.latestDataset?.image_count ? `${overviewStats.latestDataset.image_count} images` : 'Upload folders to begin'}</div>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                      <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Latest Model</div>
                      <div className="mt-2 text-sm text-white">{overviewStats.latestModel?.model_id ? overviewStats.latestModel.model_id.slice(0, 8) : 'None yet'}</div>
                      <div className="mt-1 text-xs text-slate-500">{overviewStats.latestModel?.metrics?.accuracy ? `Accuracy ${(overviewStats.latestModel.metrics.accuracy * 100).toFixed(1)}%` : 'Train a model to populate this card'}</div>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                      <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Latest Job</div>
                      <div className="mt-2 text-sm text-white">{overviewStats.latestJob?.status || 'None yet'}</div>
                      <div className="mt-1 text-xs text-slate-500">{overviewStats.latestJob?.progress ? `${overviewStats.latestJob.progress.toFixed(0)}% complete` : 'No training job active'}</div>
                    </div>
                  </div>
                </div>

                <div className="glass-panel p-6">
                  <SectionHeader title="Quick Access" subtitle="Fast actions for the core workflow." />
                  <div className="space-y-3">
                    <button className="button-primary w-full" onClick={() => setActivePage('dataset')}>Upload Dataset</button>
                    <button className="button-secondary w-full" onClick={() => setActivePage('train')}>Start Training</button>
                    <button className="button-secondary w-full" onClick={() => setActivePage('prediction')}>Run Prediction</button>
                    <button className="button-secondary w-full" onClick={() => setActivePage('statistics')}>View Statistics</button>
                  </div>
                </div>
              </section>
            </div>
          ) : null}

          {activePage === 'dataset' ? (
            <div className="mt-6 grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
              <div className="glass-panel p-6">
                <SectionHeader title="Dataset Upload" subtitle="Upload one folder of real images and one folder of fake images. Labels are assigned automatically from the folder you pick." />

                <form onSubmit={handleDatasetUpload} className="space-y-5">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <label className="label-text">Real Folder</label>
                      <input type="file" multiple accept="image/*" webkitdirectory="" directory="" className="input-field py-2.5" onChange={(event) => handleFolderSelection(event, setRealFolderFiles)} />
                      <div className="mt-2 text-xs text-slate-400">Pick the folder that contains real images, for example <span className="text-slate-200">train/real</span>.</div>
                    </div>
                    <div>
                      <label className="label-text">Fake Folder</label>
                      <input type="file" multiple accept="image/*" webkitdirectory="" directory="" className="input-field py-2.5" onChange={(event) => handleFolderSelection(event, setFakeFolderFiles)} />
                      <div className="mt-2 text-xs text-slate-400">Pick the folder that contains fake images, for example <span className="text-slate-200">train/fake</span>.</div>
                    </div>
                  </div>

                  {(realFolderFiles.length > 0 || fakeFolderFiles.length > 0) ? (
                    <div className="grid gap-4 md:grid-cols-2">
                      <FolderSummary title="Real folder" files={realFolderFiles} />
                      <FolderSummary title="Fake folder" files={fakeFolderFiles} />
                    </div>
                  ) : null}

                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <label className="label-text">Generator Type</label>
                      <input className="input-field" value={datasetGeneratorType} onChange={(event) => setDatasetGeneratorType(event.target.value)} placeholder="CIFAKE" />
                    </div>
                    <div>
                      <label className="label-text">Image Size</label>
                      <input className="input-field" value={datasetImageSize} onChange={(event) => setDatasetImageSize(event.target.value)} placeholder="128,128" />
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-3">
                    <button type="submit" className="button-primary" disabled={uploadingDataset}>{uploadingDataset ? 'Uploading...' : 'Upload Dataset'}</button>
                    <button type="button" className="button-secondary" onClick={refreshOverview}>Refresh State</button>
                  </div>
                </form>

                {datasetMessage ? <div className="mt-5 rounded-2xl border border-cyan-400/20 bg-cyan-400/10 px-4 py-3 text-sm text-cyan-50">{datasetMessage}</div> : null}
              </div>

              <div className="glass-panel p-6">
                <SectionHeader title="Registered Datasets" subtitle="The backend dataset_db is reflected here after each upload." />
                <div className="space-y-3">
                  {activeDatasetEntries.length === 0 ? (
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-400">No datasets are registered yet.</div>
                  ) : (
                    activeDatasetEntries.map(([id, dataset]) => (
                      <div key={id} className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium text-white">{dataset.name}</div>
                            <div className="mt-1 text-xs text-slate-400">{id}</div>
                          </div>
                          <span className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs text-cyan-200">{dataset.status}</span>
                        </div>
                        <div className="mt-4 grid grid-cols-2 gap-3 text-sm text-slate-300">
                          <div>Images: {dataset.image_count}</div>
                          <div>Real: {dataset.real_count ?? 'N/A'}</div>
                          <div>Fake: {dataset.fake_count ?? 'N/A'}</div>
                          <div>Type: {dataset.generator_type}</div>
                          <div>Size: {clampText(Array.isArray(dataset.img_size) ? dataset.img_size.join('x') : dataset.img_size)}</div>
                          <div>Uploaded: {formatDate(dataset.uploaded_at)}</div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          ) : null}

          {activePage === 'train' ? (
            <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_0.9fr]">
              <div className="glass-panel p-6">
                <SectionHeader title="Train Model" subtitle="Trigger the notebook-derived background thread, which updates training_jobs_db while the UI keeps polling for progress." />
                <form onSubmit={handleStartTraining} className="space-y-5">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <label className="label-text">Dataset ID</label>
                      <select className="input-field" value={trainDatasetId} onChange={(event) => setTrainDatasetId(event.target.value)}>
                        <option value="" className="bg-slate-950">Select dataset</option>
                        {activeDatasetEntries.map(([id, dataset]) => (
                          <option key={id} value={id} className="bg-slate-950">{dataset.name} - {id.slice(0, 8)}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="label-text">Epochs</label>
                      <input className="input-field" type="number" min="1" value={trainEpochs} onChange={(event) => setTrainEpochs(event.target.value)} />
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-3">
                    <button type="submit" className="button-primary" disabled={startingTraining}>{startingTraining ? 'Starting...' : 'Start Training'}</button>
                    <button type="button" className="button-secondary" onClick={() => setActivePage('monitor')}>Open Monitor</button>
                  </div>
                </form>

                {trainingMessage ? <div className="mt-5 rounded-2xl border border-cyan-400/20 bg-cyan-400/10 px-4 py-3 text-sm text-cyan-50">{trainingMessage}</div> : null}
              </div>

              <div className="glass-panel p-6">
                <SectionHeader title="Available Models" subtitle="Completed training jobs are stored in models_db and saved locally under backend/saved_models." />
                <div className="space-y-3">
                  {activeModelEntries.length === 0 ? (
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-400">No trained models yet.</div>
                  ) : (
                    activeModelEntries.map(([id, model]) => (
                      <div key={id} className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                        <div className="text-sm font-medium text-white">{id}</div>
                        <div className="mt-2 grid gap-2 text-sm text-slate-300">
                          <div>Dataset: {model.dataset_id}</div>
                          <div>Accuracy: {model.metrics?.accuracy ? `${(model.metrics.accuracy * 100).toFixed(1)}%` : 'N/A'}</div>
                          <div>Saved: {clampText(model.saved_path)}</div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          ) : null}

          {activePage === 'monitor' ? (
            <div className="mt-6 space-y-6">
              <div className="glass-panel p-6">
                <SectionHeader
                  title="Training Monitor"
                  subtitle="Poll the job status endpoint, plot the metrics history, and inspect logs from the background thread."
                  action={
                    <div className="flex gap-3">
                      <select className="input-field w-64" value={selectedMonitorJobId} onChange={(event) => setSelectedMonitorJobId(event.target.value)}>
                        <option value="" className="bg-slate-950">Select job</option>
                        {activeJobEntries.map(([id, job]) => (
                          <option key={id} value={id} className="bg-slate-950">{job.status} - {id.slice(0, 8)}</option>
                        ))}
                      </select>
                      <button className="button-secondary" onClick={refreshMonitorNow}>Refresh Now</button>
                    </div>
                  }
                />

                {monitorError ? <div className="mb-4 rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">{monitorError}</div> : null}

                {monitorSnapshot ? (
                  <div className="space-y-6">
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <MetricCard title="Status" value={clampText(monitorSnapshot.status)} helper={`Dataset ${clampText(monitorSnapshot.dataset_id)}`} />
                      <MetricCard title="Current Epoch" value={clampText(monitorSnapshot.current_epoch ?? 0)} helper={`Target epochs ${clampText(monitorSnapshot.epochs)}`} />
                      <MetricCard title="Progress" value={`${Number(monitorSnapshot.progress || 0).toFixed(0)}%`} helper="Background thread updates this in real time" />
                      <MetricCard title="Model ID" value={monitorSnapshot.model_id ? monitorSnapshot.model_id.slice(0, 8) : 'Pending'} helper="Populated when training completes" />
                    </div>

                    <div className="grid gap-6 xl:grid-cols-[1.4fr_0.6fr]">
                      <div className="rounded-3xl border border-white/10 bg-slate-950/60 p-4" style={{ minHeight: 380 }}>
                        <Suspense fallback={<div className="flex h-80 items-center justify-center rounded-2xl border border-dashed border-white/10 text-sm text-slate-500">Loading training chart...</div>}>
                          <MonitorLineChart data={monitorChartData} options={chartOptions} />
                        </Suspense>
                      </div>

                      <div className="space-y-4">
                        <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                          <div className="text-sm font-medium text-white">Current Metrics</div>
                          <div className="mt-4 grid gap-3 text-sm text-slate-300">
                            <div>Loss: {monitorSnapshot.metrics?.loss?.toFixed ? monitorSnapshot.metrics.loss.toFixed(4) : 'N/A'}</div>
                            <div>Accuracy: {monitorSnapshot.metrics?.accuracy?.toFixed ? monitorSnapshot.metrics.accuracy.toFixed(4) : 'N/A'}</div>
                            <div>Val Loss: {monitorSnapshot.metrics?.val_loss?.toFixed ? monitorSnapshot.metrics.val_loss.toFixed(4) : 'N/A'}</div>
                            <div>Val Accuracy: {monitorSnapshot.metrics?.val_accuracy?.toFixed ? monitorSnapshot.metrics.val_accuracy.toFixed(4) : 'N/A'}</div>
                          </div>
                        </div>

                        <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                          <div className="text-sm font-medium text-white">Training Logs</div>
                          <div className="mt-4 max-h-56 space-y-2 overflow-auto text-sm text-slate-300">
                            {(monitorSnapshot.logs || []).length === 0 ? (
                              <div className="text-slate-500">No log entries yet.</div>
                            ) : (
                              monitorSnapshot.logs.map((entry, index) => (
                                <div key={`${entry}-${index}`} className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">{entry}</div>
                              ))
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-400">Select a job to start polling its status.</div>
                )}
              </div>
            </div>
          ) : null}

          {activePage === 'prediction' ? (
            <div className="mt-6 grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
              <div className="glass-panel p-6">
                <SectionHeader title="Prediction" subtitle="Upload a single image, run it through the notebook-derived endpoint, and inspect the prediction label with the FFT visualization." />
                <form onSubmit={handlePrediction} className="space-y-5">
                  <div>
                    <label className="label-text">Model</label>
                    <select className="input-field" value={predictionModelId} onChange={(event) => setPredictionModelId(event.target.value)}>
                      <option value="" className="bg-slate-950">Select model</option>
                      {activeModelEntries.map(([id, model]) => (
                        <option key={id} value={id} className="bg-slate-950">{id.slice(0, 8)} - {model.metrics?.accuracy ? `${(model.metrics.accuracy * 100).toFixed(1)}%` : 'No metric'}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="label-text">Image</label>
                    <input className="input-field py-2.5" type="file" accept="image/*" onChange={(event) => setPredictionFile(event.target.files?.[0] || null)} />
                  </div>
                  <button type="submit" className="button-primary" disabled={runningPrediction}>{runningPrediction ? 'Predicting...' : 'Run Prediction'}</button>
                </form>
                {predictionMessage ? <div className="mt-5 rounded-2xl border border-cyan-400/20 bg-cyan-400/10 px-4 py-3 text-sm text-cyan-50">{predictionMessage}</div> : null}

                {predictionResult?.status === 'success' ? (
                  <div className="mt-5 grid gap-3 md:grid-cols-3">
                    <MetricCard title="Prediction Label" value={predictionResult.prediction} helper="Notebook-compatible class output" />
                    <MetricCard title="Confidence" value={`${(predictionResult.confidence * 100).toFixed(2)}%`} helper="Returned by the backend" />
                    <MetricCard title="Probability" value={`${(predictionResult.probability * 100).toFixed(2)}%`} helper="Mirrors the confidence value" />
                  </div>
                ) : null}
              </div>

              <div className="space-y-6">
                <div className="glass-panel p-6">
                  <SectionHeader title="Preview & FFT" subtitle="The left pane shows the uploaded image while the right pane renders the FFT magnitude spectrum." />
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                      <div className="mb-3 text-sm font-medium text-white">Uploaded Image</div>
                      {predictionPreview ? <img src={predictionPreview} alt="Uploaded preview" className="h-80 w-full rounded-2xl object-cover" /> : <div className="flex h-80 items-center justify-center rounded-2xl border border-dashed border-white/10 text-sm text-slate-500">Select an image to preview it here.</div>}
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-4" style={{ minHeight: 360 }}>
                      <div className="mb-3 text-sm font-medium text-white">FFT Visualization</div>
                      {predictionResult?.fft_visualization ? (
                        <Suspense fallback={<div className="flex h-80 items-center justify-center rounded-2xl border border-dashed border-white/10 text-sm text-slate-500">Loading FFT chart...</div>}>
                          <PlotlyChart data={predictionFFTPlot} layout={{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', margin: { l: 30, r: 20, t: 20, b: 20 }, autosize: true, height: 320, font: { color: '#cbd5e1' } }} style={{ width: '100%', height: '320px' }} config={{ displayModeBar: false, responsive: true }} />
                        </Suspense>
                      ) : (
                        <div className="flex h-80 items-center justify-center rounded-2xl border border-dashed border-white/10 text-sm text-slate-500">FFT visualization will appear after a prediction completes.</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          {activePage === 'statistics' ? (
            <div className="mt-6 space-y-6">
              <div className="glass-panel p-6">
                <SectionHeader
                  title="Statistics"
                  subtitle="Load ROC, AUC, precision, recall, F1, box plot, and Mann-Whitney test results for any trained model."
                  action={
                    <div className="flex flex-wrap gap-3">
                      <select className="input-field w-72" value={statisticsModelId} onChange={(event) => setStatisticsModelId(event.target.value)}>
                        <option value="" className="bg-slate-950">Select model</option>
                        {activeModelEntries.map(([id, model]) => (
                          <option key={id} value={id} className="bg-slate-950">{id.slice(0, 8)} - {model.metrics?.accuracy ? `${(model.metrics.accuracy * 100).toFixed(1)}%` : 'No metric'}</option>
                        ))}
                      </select>
                      <button className="button-primary" onClick={handleStatisticsLoad} disabled={loadingStatistics}>{loadingStatistics ? 'Loading...' : 'Load Statistics'}</button>
                    </div>
                  }
                />
                {statisticsMessage ? <div className="mb-5 rounded-2xl border border-cyan-400/20 bg-cyan-400/10 px-4 py-3 text-sm text-cyan-50">{statisticsMessage}</div> : null}

                {statisticsResult ? (
                  <div className="space-y-6">
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <MetricCard title="AUC" value={statisticsResult.roc_curve?.auc?.toFixed ? statisticsResult.roc_curve.auc.toFixed(4) : 'N/A'} helper="Area under ROC curve" />
                      <MetricCard title="Precision" value={statisticsResult.metrics?.precision?.toFixed ? statisticsResult.metrics.precision.toFixed(4) : 'N/A'} helper="Positive predictive value" />
                      <MetricCard title="Recall" value={statisticsResult.metrics?.recall?.toFixed ? statisticsResult.metrics.recall.toFixed(4) : 'N/A'} helper="Sensitivity" />
                      <MetricCard title="F1 Score" value={statisticsResult.metrics?.f1_score?.toFixed ? statisticsResult.metrics.f1_score.toFixed(4) : 'N/A'} helper="Harmonic mean" />
                    </div>

                    <div className="grid gap-6 xl:grid-cols-2">
                      <div className="rounded-3xl border border-white/10 bg-slate-950/60 p-4" style={{ minHeight: 380 }}>
                        <div className="mb-3 text-sm font-medium text-white">ROC Curve</div>
                        <Suspense fallback={<div className="flex h-80 items-center justify-center rounded-2xl border border-dashed border-white/10 text-sm text-slate-500">Loading ROC chart...</div>}>
                          <PlotlyChart data={statisticsROCPlot} layout={{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', margin: { l: 40, r: 20, t: 20, b: 40 }, xaxis: { title: 'False Positive Rate', gridcolor: 'rgba(148,163,184,0.12)', color: '#cbd5e1' }, yaxis: { title: 'True Positive Rate', gridcolor: 'rgba(148,163,184,0.12)', color: '#cbd5e1' }, font: { color: '#cbd5e1' }, autosize: true, height: 340 }} style={{ width: '100%', height: '340px' }} config={{ displayModeBar: false, responsive: true }} />
                        </Suspense>
                      </div>

                      <div className="rounded-3xl border border-white/10 bg-slate-950/60 p-4" style={{ minHeight: 380 }}>
                        <div className="mb-3 text-sm font-medium text-white">Box Plot</div>
                        <Suspense fallback={<div className="flex h-80 items-center justify-center rounded-2xl border border-dashed border-white/10 text-sm text-slate-500">Loading box plot...</div>}>
                          <PlotlyChart data={statisticsBoxPlot} layout={{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', margin: { l: 40, r: 20, t: 20, b: 40 }, yaxis: { title: 'Normalized Energy', gridcolor: 'rgba(148,163,184,0.12)', color: '#cbd5e1' }, font: { color: '#cbd5e1' }, autosize: true, height: 340 }} style={{ width: '100%', height: '340px' }} config={{ displayModeBar: false, responsive: true }} />
                        </Suspense>
                      </div>
                    </div>

                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                      <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-5">
                        <div className="text-sm font-medium text-white">Mann-Whitney Test</div>
                        <div className="mt-3 text-sm text-slate-300">Test: {statisticsResult.mann_whitney?.test || 'N/A'}</div>
                        <div className="mt-2 text-sm text-slate-300">p-value: {statisticsResult.mann_whitney?.p_value?.toFixed ? statisticsResult.mann_whitney.p_value.toFixed(6) : 'N/A'}</div>
                      </div>
                      <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-5">
                        <div className="text-sm font-medium text-white">Model Summary</div>
                        <div className="mt-3 text-sm text-slate-300">Model ID: {statisticsResult.model_summary?.model_id || 'N/A'}</div>
                        <div className="mt-2 text-sm text-slate-300">Dataset: {statisticsResult.model_summary?.dataset_id || 'N/A'}</div>
                      </div>
                      <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-5">
                        <div className="text-sm font-medium text-white">Notes</div>
                        <div className="mt-3 text-sm leading-6 text-slate-300">The statistics endpoint preserves the notebook’s rpy2 integration path while also returning browser-ready data for the React charts.</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-400">Select a trained model and load statistics to see the charts.</div>
                )}
              </div>
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}

export default App;
