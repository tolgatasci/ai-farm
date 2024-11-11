
const { useState, useEffect, useRef } = React;

function TaskForm({ onSubmit }) {

    const formRef = useRef(null);
       const [task, setTask] = useState({
        type: 'training',
        name: 'sonnet',
        url: 'http://localhost:5000/models/sonnet/1.0',
        version: '1.0',
        batch_size: 32,
        learning_rate: 0.001,
        epochs: 10,
        distributed: false,
        n_clients: 3,
        aggregation_frequency: 5,
        warmup_epochs: 2,
        requirements: {
            min_gpu_memory: 0,
            min_cpu_cores: 1,
            min_ram: 1
        },
        priority: 0,
        data_config: {
            split_method: 'random',
            validation_ratio: 0.2
        }
    });

    // Form submission state
    const [submitting, setSubmitting] = useState(false);
    // Task tipleri
    const taskTypes = [
        { value: 'training', label: 'Model Training' },
        { value: 'fine-tuning', label: 'Fine Tuning' },
        { value: 'evaluation', label: 'Model Evaluation' },
        { value: 'inference', label: 'Inference' },
        { value: 'data-processing', label: 'Data Processing' }
    ];

    const handleSubmit = async (e) => {
        e.preventDefault();
        e.stopPropagation();

        if(submitting) return;

        try {
            setSubmitting(true);
            await onSubmit({...task});
            // Form submit başarılı olduktan sonra
            if(formRef.current) {
                formRef.current.reset();
            }
        } catch(error) {
            console.error('Form submission error:', error);
        } finally {
            setSubmitting(false);
        }

        return false;
    };

    return (
        <form
            ref={formRef}
            onSubmit={handleSubmit}
            className="mb-8 p-4 border rounded"
            noValidate // Browser validasyonunu devre dışı bırak
        >
            <h3 className="text-xl font-semibold mb-4">Create New Task</h3>
            <div className="space-y-4">
                <div>
                    <label className="block text-sm font-medium mb-1">Task Type</label>
                    <select
                        value={task.type}
                        onChange={(e) => setTask({...task, type: e.target.value})}
                        className="w-full p-2 border rounded bg-white"
                        required
                    >
                        {taskTypes.map(type => (
                            <option key={type.value} value={type.value}>
                                {type.label}
                            </option>
                        ))}
                    </select>

                    {/* Task tipine göre açıklama göster */}
                    <div className="mt-1 text-sm text-gray-500">
                        {task.type === 'training' && 'Train a new model from scratch'}
                        {task.type === 'fine-tuning' && 'Fine-tune an existing model'}
                        {task.type === 'evaluation' && 'Evaluate model performance'}
                        {task.type === 'inference' && 'Run model inference'}
                        {task.type === 'data-processing' && 'Process and prepare training data'}
                    </div>
                </div>

                {/* Task tipine göre form alanlarını göster/gizle */}
                {(task.type === 'training' || task.type === 'fine-tuning') && (
                    <>
                        <div>
                            <label className="block text-sm font-medium mb-1">Model Name</label>
                            <input
                                type="text"
                                value={task.name}
                                onChange={(e) => setTask({...task, name: e.target.value})}
                                className="w-full p-2 border rounded"
                                placeholder="mnist_classifier"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Model URL</label>
                            <input
                                type="text"
                                value={task.url}
                                onChange={(e) => setTask({...task, url: e.target.value})}
                                className="w-full p-2 border rounded"
                                placeholder="http://model-repo/mnist/v1"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Model Version</label>
                            <input
                                type="text"
                                value={task.version}
                                onChange={(e) => setTask({...task, version: e.target.value})}
                                className="w-full p-2 border rounded"
                                placeholder="1.0"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Batch Size</label>
                            <input
                                type="number"
                                value={task.batch_size}
                                onChange={(e) => setTask({...task, batch_size: parseInt(e.target.value)})}
                                className="w-full p-2 border rounded"
                                min="1"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Learning Rate</label>
                            <input
                                type="number"
                                value={task.learning_rate}
                                onChange={(e) => setTask({...task, learning_rate: parseFloat(e.target.value)})}
                                className="w-full p-2 border rounded"
                                step="0.0001"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Epochs</label>
                            <input
                                type="number"
                                value={task.epochs}
                                onChange={(e) => setTask({...task, epochs: parseInt(e.target.value)})}
                                className="w-full p-2 border rounded"
                                min="1"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">
                                <input
                                    type="checkbox"
                                    checked={task.distributed}
                                    onChange={(e) => setTask({...task, distributed: e.target.checked})}
                                    className="mr-2"
                                />
                                Distributed Training
                            </label>
                        </div>
                    </>
                )}
                {/* Distributed Training seçildiğinde gösterilecek alanlar */}
                {task.distributed && (
                    <div className="space-y-4 mt-4 p-4 bg-gray-50 rounded">
                        <h4 className="font-medium text-gray-700">Parallel Training Configuration</h4>

                        <div>
                            <label className="block text-sm font-medium mb-1">Number of Clients</label>
                            <input
                                type="number"
                                value={task.n_clients}
                                onChange={(e) => setTask({...task, n_clients: parseInt(e.target.value)})}
                                className="w-full p-2 border rounded"
                                min="2"
                                required
                            />
                            <p className="text-xs text-gray-500 mt-1">Minimum 2 clients required for distributed
                                training</p>
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Aggregation Frequency (epochs)</label>
                            <input
                                type="number"
                                value={task.aggregation_frequency}
                                onChange={(e) => setTask({...task, aggregation_frequency: parseInt(e.target.value)})}
                                className="w-full p-2 border rounded"
                                min="1"
                                required
                            />
                            <p className="text-xs text-gray-500 mt-1">How often to aggregate model updates from
                                clients</p>
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Warmup Epochs</label>
                            <input
                                type="number"
                                value={task.warmup_epochs}
                                onChange={(e) => setTask({...task, warmup_epochs: parseInt(e.target.value)})}
                                className="w-full p-2 border rounded"
                                min="0"
                                required
                            />
                            <p className="text-xs text-gray-500 mt-1">Number of epochs for initial local training</p>
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Data Split Method</label>
                            <select
                                value={task.data_config.split_method}
                                onChange={(e) => setTask({
                                    ...task,
                                    data_config: {...task.data_config, split_method: e.target.value}
                                })}
                                className="w-full p-2 border rounded bg-white"
                                required
                            >
                                <option value="random">Random Split</option>
                                <option value="sequential">Sequential Split</option>
                                <option value="custom">Custom Split</option>
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Validation Split Ratio</label>
                            <input
                                type="number"
                                value={task.data_config.validation_ratio}
                                onChange={(e) => setTask({
                                    ...task,
                                    data_config: {...task.data_config, validation_ratio: parseFloat(e.target.value)}
                                })}
                                className="w-full p-2 border rounded"
                                step="0.1"
                                min="0"
                                max="0.5"
                                required
                            />
                        </div>
                    </div>
                )}
                {/* Her task tipi için requirements göster */}
                <div>
                    <label className="block text-sm font-medium mb-1">Requirements</label>
                    <div className="space-y-2">
                        <div>
                            <label className="text-xs text-gray-600">Min GPU Memory (GB)</label>
                            <input
                                type="number"
                                value={task.requirements.min_gpu_memory}
                                onChange={(e) => setTask({
                                    ...task,
                                    requirements: {
                                        ...task.requirements,
                                        min_gpu_memory: parseFloat(e.target.value)
                                    }
                                })}
                                className="w-full p-2 border rounded"
                                min="0"
                                step="0.5"
                            />
                        </div>
                        <div>
                            <label className="text-xs text-gray-600">Min CPU Cores</label>
                            <input
                                type="number"
                                value={task.requirements.min_cpu_cores}
                                onChange={(e) => setTask({
                                    ...task,
                                    requirements: {
                                        ...task.requirements,
                                        min_cpu_cores: parseInt(e.target.value)
                                    }
                                })}
                                className="w-full p-2 border rounded"
                                min="1"
                            />
                        </div>
                        <div>
                            <label className="text-xs text-gray-600">Min RAM (GB)</label>
                            <input
                                type="number"
                                value={task.requirements.min_ram}
                                onChange={(e) => setTask({
                                    ...task,
                                    requirements: {
                                        ...task.requirements,
                                        min_ram: parseFloat(e.target.value)
                                    }
                                })}
                                className="w-full p-2 border rounded"
                                min="1"
                                step="0.5"
                            />
                        </div>
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium mb-1">Priority</label>
                    <input
                        type="number"
                        value={task.priority}
                        onChange={(e) => setTask({...task, priority: parseInt(e.target.value)})}
                        className="w-full p-2 border rounded"
                        min="0"
                        required
                    />
                </div>

                <button
                    type="submit"
                    disabled={submitting}
                    className={`w-full ${submitting ? 'bg-gray-500' : 'bg-blue-500 hover:bg-blue-600'} text-white py-2 rounded`}
                >
                    {submitting ? 'Creating Task...' : 'Create Task'}
                </button>
            </div>
        </form>
    );
}

// TaskList bileşenini güncelleyin
function TaskList({tasks}) {
    const [taskStates, setTaskStates] = useState({});
    const [showForm, setShowForm] = useState(false);

    useEffect(() => {
        const newTaskStates = {};
        if (Array.isArray(tasks)) {
            tasks.forEach(task => {
        const config = typeof task.config === 'string' ? JSON.parse(task.config) : task.config;
        const result = typeof task.result === 'string' ? JSON.parse(task.result) : task.result;

        const activeClients = task.assignments?.filter(
          a => ['assigned', 'training'].includes(a.status)
        )?.length || 0;

        const progress = result?.progress || 0;

        newTaskStates[task.id] = {
          activeClients,
          progress,
          distributed: config?.distributed || false,
          targetClients: config?.n_clients || 1
        };
      });
    }
    setTaskStates(newTaskStates);
  }, [tasks]);

  if (!Array.isArray(tasks) || tasks.length === 0) {
    return <div className="text-center py-4">No tasks available</div>;
  }

  const deleteTask = async (taskId) => {
    try {
      const response = await fetch(`/api/tasks/${taskId}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error('Failed to delete task');
      // TaskList yenilenecektir çünkü parent component datayı güncelleyecek
    } catch (error) {
      console.error('Error deleting task:', error);
      alert('Failed to delete task');
    }
  };

  const updateTaskStatus = async (taskId, newStatus) => {
    try {
      const response = await fetch(`/api/tasks/${taskId}/status`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      });
      if (!response.ok) throw new Error('Failed to update task status');
      // TaskList yenilenecektir çünkü parent component datayı güncelleyecek
    } catch (error) {
      console.error('Error updating task status:', error);
      alert('Failed to update task status');
    }
  };
    const formatProgress = (progress) => {
    if (!progress) return '0%';
    const value = parseFloat(progress);
    return `${Math.round(value)}%`;
  };
  if (!Array.isArray(tasks) || tasks.length === 0) {
        return <div className="text-center py-4">No tasks available</div>;
    }

  return (
    <div className="overflow-x-auto">
      

      {showForm && <TaskForm onSubmit={(task) => {
        handleCreateTask(task);
        setShowForm(false);
      }} />}

      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Priority</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Distributed</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Active Clients</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Progress</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {tasks.map(task => {
            const taskState = taskStates[task.id] || {};
            return (
              <tr key={task?.id || Math.random()}>
                    <td className="px-6 py-4 whitespace-nowrap">{task.id}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{task.type}</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                        <select
                            value={task.status || ''}  // Eğer status undefined olursa boş string kullan
                            onChange={(e) => updateTaskStatus(task.id, e.target.value)}
                            className={`px-2 py-1 text-sm rounded-full border ${
                                task.status === 'completed' ? 'bg-green-100 text-green-800 border-green-200' :
                                    task.status === 'failed' ? 'bg-red-100 text-red-800 border-red-200' :
                                        task.status === 'running' ? 'bg-blue-100 text-blue-800 border-blue-200' :
                                            'bg-yellow-100 text-yellow-800 border-yellow-200'
                            }`}
                        >
                            <option value="pending">Pending</option>
                            <option value="running">Running</option>
                            <option value="completed">Completed</option>
                            <option value="failed">Failed</option>
                        </select>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">{task.priority}</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                        {taskState.distributed ? (
                            <span className="text-green-600">
                      Yes ({taskState.targetClients} clients)
                    </span>
                        ) : (
                            <span className="text-gray-500">No</span>
                        )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                        {taskState.distributed ? (
                            <span className={taskState.activeClients === taskState.targetClients ?
                                'text-green-600' : 'text-yellow-600'}>
                      {taskState.activeClients}/{taskState.targetClients}
                    </span>
                        ) : '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div
                                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                                style={{width: `${taskState.progress || 0}%`}}
                            />
                        </div>
                        <span className="text-xs text-gray-500 mt-1">
                      {formatProgress(taskState.progress)}

                  </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                        <button
                            onClick={() => deleteTask(task.id)}
                            className="text-red-600 hover:text-red-900"
                        >
                            Delete
                        </button>
                    </td>
                </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ClientList({clients}) {
  if (!Array.isArray(clients) || clients.length === 0) {
        return <div className="text-center py-4">No clients connected</div>;
    }


    return (
        <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">CPU</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">RAM</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">GPU</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Last Seen</th>
                    </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                    {clients.map(client => (
                        <tr key={client?.id || Math.random()}>
                            <td className="px-6 py-4 whitespace-nowrap">{client.id}</td>
                            <td className="px-6 py-4 whitespace-nowrap">
                                <span className={`px-2 py-1 text-xs rounded-full 
                                    ${client.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                                    {client.status}
                                </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                                {client.cpu_info?.usage_percent?.toFixed(1)}%
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                                {client.ram_info?.percent?.toFixed(1)}%
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                                {client.gpu_info ?
                                    Array.isArray(client.gpu_info) ?
                                        client.gpu_info.map(gpu => gpu.name).join(', ') :
                                        'No GPU info'
                                    : 'No GPU'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                                {new Date(client.last_seen).toLocaleString()}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function AdminPanel() {
    const [clients, setClients] = useState([]);
    const [tasks, setTasks] = useState([]);
    const [activeTab, setActiveTab] = useState('clients');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showTaskForm, setShowTaskForm] = useState(false);
    const refreshInterval = useRef(null);
    async function fetchData() {
        try {
            setLoading(true);
            const [clientsRes, tasksRes] = await Promise.all([
                fetch('/api/clients'),
                fetch('/api/tasks')
            ]);

            if (!clientsRes.ok || !tasksRes.ok) {
                throw new Error('Failed to fetch data');
            }

            const [clientsData, tasksData] = await Promise.all([
                clientsRes.json(),
                tasksRes.json()
            ]);

            setClients(clientsData || []);
            setTasks(tasksData || []);
            setError(null);
        } catch (err) {
            setError('Failed to load data. Please try again later.');
            console.error('Error fetching data:', err);
            setClients([]);
            setTasks([]);
        } finally {
            setLoading(false);
        }
    }

  const handleCreateTask = async (taskData) => {
        try {
            const response = await fetch('/api/tasks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(taskData),
            });

            if (!response.ok) {
                throw new Error('Failed to create task');
            }

            await fetchData(); // Veriyi güncelle
            setShowTaskForm(false); // Formu kapat

        } catch (err) {
            console.error('Error creating task:', err);
            alert('Failed to create task. Please try again.');
        }
    };


   useEffect(() => {
        // İlk yükleme
        fetchData();

        // Task oluşturma formu açıkken yenilemeyi durdur
        if (!showTaskForm) {
            refreshInterval.current = setInterval(fetchData, 10000);
        }

        // Cleanup function
        return () => {
            if (refreshInterval.current) {
                clearInterval(refreshInterval.current);
            }
        };
    }, [showTaskForm]); // showTaskForm değiştiğinde effect'i yeniden çalıştır
    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                    <strong className="font-bold">Error: </strong>
                    <span className="block sm:inline">{error}</span>
                    <button onClick={fetchData} className="mt-2 bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600">
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="container mx-auto px-4 py-8">
            <div className="mb-8">
                <h1 className="text-3xl font-bold mb-4">AI Farm Admin Panel</h1>
                <div className="space-x-4">
                    <button
                        onClick={() => setActiveTab('clients')}
                        className={`px-4 py-2 rounded transition ${
                            activeTab === 'clients' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'
                        }`}
                    >
                        Clients ({Array.isArray(clients) ? clients.length : 0})
                    </button>
                    <button
                        onClick={() => setActiveTab('tasks')}
                        className={`px-4 py-2 rounded transition ${
                            activeTab === 'tasks' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'
                        }`}
                    >
                        Tasks ({Array.isArray(tasks) ? tasks.length : 0})
                    </button>
                </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
                {activeTab === 'clients' ? (
                    <>
                        <h2 className="text-2xl font-semibold mb-4">Connected Clients</h2>
                        <ClientList clients={Array.isArray(clients) ? clients : []} />
                    </>
                ) : (
                    <>
                        <h2 className="text-2xl font-semibold mb-4">Task Management</h2>
                        <div className="mb-4">
                            <button
                                onClick={() => setShowTaskForm(!showTaskForm)}
                                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                            >
                                {showTaskForm ? 'Hide Form' : 'New Task'}
                            </button>
                        </div>
                     {showTaskForm && (
            <TaskForm
                onSubmit={handleCreateTask}
            />
        )}
                        <TaskList tasks={Array.isArray(tasks) ? tasks : []} />
                    </>
                )}
            </div>
        </div>
    );
}

ReactDOM.render(<AdminPanel />, document.getElementById('root'));