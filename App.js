import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Button,
  Box,
  LinearProgress,
  Paper,
  TextField,
  Alert,
  Snackbar,
  Card,
  CardContent,
  Grid,
  IconButton,
  Tooltip,
  Divider,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  useTheme,
  alpha,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  PlayArrow as StartIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
  Code as CodeIcon,
  Settings as SettingsIcon,
  Dashboard as DashboardIcon,
  AutoFixHigh as AutoFixIcon,
} from '@mui/icons-material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#0a1929',
      paper: '#132f4c',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
          padding: '8px 16px',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundImage: 'none',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
  },
});

function App() {
  const [dataFile, setDataFile] = useState(null);
  const [config, setConfig] = useState('');
  const [status, setStatus] = useState('idle');
  const [losses, setLosses] = useState([]);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [code, setCode] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');

  useEffect(() => {
    // Load sample config on mount
    fetch('http://localhost:8000/')
      .then(() => {
        const sampleConfig = {
          base_model: "gpt2",
          style_dim: 5,
          pattern_dim: 3,
          max_length: 32,
          batch_size: 1,
          num_workers: 0,
          learning_rate: 1e-5,
          num_epochs: 1,
          device: "cpu"
        };
        setConfig(JSON.stringify(sampleConfig, null, 2));
      })
      .catch(() => {
        setError('Backend server is not running. Please start the FastAPI server.');
      });
  }, []);

  const showSnackbar = (message, severity = 'success') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleDataChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setDataFile(file);
      showSnackbar('File selected: ' + file.name);
    }
  };

  const handleConfigChange = (e) => {
    setConfig(e.target.value);
  };

  const handleCodeChange = (e) => {
    setCode(e.target.value);
  };

  const uploadData = async () => {
    if (!dataFile) {
      showSnackbar('Please select a file first', 'error');
      return;
    }
    try {
      const formData = new FormData();
      formData.append('file', dataFile);
      const response = await fetch('http://localhost:8000/upload-data/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        showSnackbar(data.message);
      } else {
        throw new Error(data.detail || 'Upload failed');
      }
    } catch (err) {
      showSnackbar(err.message, 'error');
    }
  };

  const uploadConfig = async () => {
    if (!config) {
      showSnackbar('Please enter config data', 'error');
      return;
    }
    try {
      const response = await fetch('http://localhost:8000/upload-config/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: config,
      });
      const data = await response.json();
      if (response.ok) {
        showSnackbar(data.message);
      } else {
        throw new Error(data.detail || 'Upload failed');
      }
    } catch (err) {
      showSnackbar(err.message, 'error');
    }
  };

  const startTraining = async () => {
    try {
      const response = await fetch('http://localhost:8000/train/', {
        method: 'POST',
      });
      const data = await response.json();
      if (response.ok) {
        setStatus('training');
        showSnackbar(data.message);
        pollStatus();
      } else {
        throw new Error(data.detail || 'Training failed to start');
      }
    } catch (err) {
      showSnackbar(err.message, 'error');
    }
  };

  const pollStatus = async () => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8000/status/');
        const data = await response.json();
        setStatus(data.status);
        setLosses(data.losses || []);
        if (data.error) {
          showSnackbar(data.error, 'error');
        }
        if (data.status === 'completed' || data.status === 'error') {
          clearInterval(interval);
          if (data.status === 'completed') {
            showSnackbar('Training completed successfully!');
          }
        }
      } catch (err) {
        showSnackbar('Failed to fetch status', 'error');
        clearInterval(interval);
      }
    }, 2000);
  };

  const downloadModel = async () => {
    setDownloading(true);
    try {
      const response = await fetch('http://localhost:8000/download-model/');
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'digital_twin_model.pth';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        showSnackbar('Model downloaded successfully');
      } else {
        const data = await response.json();
        throw new Error(data.detail || 'Download failed');
      }
    } catch (err) {
      showSnackbar(err.message, 'error');
    } finally {
      setDownloading(false);
    }
  };

  const correctCode = async (autoApply = false) => {
    try {
      const response = await fetch('http://localhost:8000/correct-code/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          language: 'python',
          auto_apply: autoApply
        }),
      });
      const data = await response.json();
      if (response.ok) {
        setSuggestions(data.suggestions);
        if (autoApply) {
          setCode(data.corrected_code);
          showSnackbar('Code automatically corrected');
        } else {
          showSnackbar('Code analysis complete');
        }
      } else {
        throw new Error(data.detail || 'Code correction failed');
      }
    } catch (err) {
      showSnackbar(err.message, 'error');
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'training': return 'primary';
      case 'completed': return 'success';
      case 'error': return 'error';
      default: return 'info';
    }
  };

  const renderDashboard = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Data Upload
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
              <Button
                variant="contained"
                component="label"
                startIcon={<UploadIcon />}
                sx={{ flexGrow: 1 }}
              >
                Select Data File
                <input type="file" hidden onChange={handleDataChange} />
              </Button>
              <Button
                variant="outlined"
                onClick={uploadData}
                disabled={!dataFile}
                startIcon={<UploadIcon />}
              >
                Upload
              </Button>
            </Box>
            {dataFile && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Selected: {dataFile.name}
              </Typography>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Configuration
            </Typography>
            <TextField
              label="Config (JSON)"
              multiline
              minRows={6}
              fullWidth
              value={config}
              onChange={handleConfigChange}
              variant="outlined"
              sx={{ mb: 2 }}
            />
            <Button
              variant="outlined"
              onClick={uploadConfig}
              disabled={!config}
              startIcon={<UploadIcon />}
              fullWidth
            >
              Upload Config
            </Button>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Training Status
            </Typography>
            <Box sx={{ mb: 2 }}>
              <Alert severity={getStatusColor()} sx={{ mb: 2 }}>
                Status: {status}
              </Alert>
              {status === 'training' && <LinearProgress />}
            </Box>
            {losses.length > 0 && (
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Training Losses:
                </Typography>
                <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                  <pre style={{ margin: 0, overflow: 'auto' }}>
                    {JSON.stringify(losses, null, 2)}
                  </pre>
                </Paper>
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={startTraining}
            disabled={status === 'training'}
            startIcon={<StartIcon />}
            sx={{ flexGrow: 1 }}
          >
            Start Training
          </Button>
          <Button
            variant="contained"
            color="success"
            onClick={downloadModel}
            disabled={downloading || status !== 'completed'}
            startIcon={<DownloadIcon />}
            sx={{ flexGrow: 1 }}
          >
            {downloading ? 'Downloading...' : 'Download Model'}
          </Button>
        </Box>
      </Grid>
    </Grid>
  );

  const renderCodeEditor = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Code Editor
            </Typography>
            <TextField
              multiline
              fullWidth
              minRows={20}
              value={code}
              onChange={handleCodeChange}
              variant="outlined"
              sx={{
                '& .MuiOutlinedInput-root': {
                  fontFamily: 'monospace',
                  fontSize: '14px',
                },
              }}
            />
            <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                color="primary"
                onClick={() => correctCode(false)}
                startIcon={<AutoFixIcon />}
              >
                Analyze Code
              </Button>
              <Button
                variant="contained"
                color="success"
                onClick={() => correctCode(true)}
                startIcon={<AutoFixIcon />}
              >
                Auto-Correct
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Suggestions
            </Typography>
            {suggestions.length > 0 ? (
              <List>
                {suggestions.map((suggestion, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <InfoIcon color={suggestion.type === 'error' ? 'error' : 'warning'} />
                    </ListItemIcon>
                    <ListItemText
                      primary={suggestion.message}
                      secondary={`Line ${suggestion.line}`}
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No suggestions yet. Click "Analyze Code" to get started.
              </Typography>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', minHeight: '100vh' }}>
        <Drawer
          variant="permanent"
          sx={{
            width: 240,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: 240,
              boxSizing: 'border-box',
              backgroundColor: theme.palette.background.paper,
              borderRight: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            },
          }}
        >
          <Box sx={{ p: 2 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Digital Twin
            </Typography>
            <List>
              <ListItem button selected={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')}>
                <ListItemIcon>
                  <DashboardIcon />
                </ListItemIcon>
                <ListItemText primary="Dashboard" />
              </ListItem>
              <ListItem button selected={activeTab === 'code'} onClick={() => setActiveTab('code')}>
                <ListItemIcon>
                  <CodeIcon />
                </ListItemIcon>
                <ListItemText primary="Code Editor" />
              </ListItem>
              <ListItem button selected={activeTab === 'settings'} onClick={() => setActiveTab('settings')}>
                <ListItemIcon>
                  <SettingsIcon />
                </ListItemIcon>
                <ListItemText primary="Settings" />
              </ListItem>
            </List>
          </Box>
        </Drawer>
        <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
          <Container maxWidth="xl">
            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}
            {activeTab === 'dashboard' && renderDashboard()}
            {activeTab === 'code' && renderCodeEditor()}
            {activeTab === 'settings' && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Settings
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Settings panel coming soon...
                  </Typography>
                </CardContent>
              </Card>
            )}
          </Container>
        </Box>
      </Box>
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App;
