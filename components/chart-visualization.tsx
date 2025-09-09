"use client"
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"

interface ChartVisualizationProps {
  chart: any
  chartType?: string
}

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8", "#82CA9D"]

export function ChartVisualization({ chart, chartType }: ChartVisualizationProps) {
  if (!chart || !chart.series || chart.series.length === 0) {
    return null
  }

  const renderChart = () => {
    const chartData = chart.series[0]?.data || []

    // Transform data for Recharts
    const transformedData = chartData.map((item: any, index: number) => ({
      name: item.x || item.name || `Item ${index + 1}`,
      value: item.y || item.value || 0,
      ...item,
    }))

    switch (chart.type) {
      case "line":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={transformedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Legend />
              {chart.series.map((series: any, index: number) => (
                <Line
                  key={index}
                  type="monotone"
                  dataKey="value"
                  stroke={COLORS[index % COLORS.length]}
                  strokeWidth={2}
                  name={series.name || `Series ${index + 1}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )

      case "bar":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={transformedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Legend />
              {chart.series.map((series: any, index: number) => (
                <Bar
                  key={index}
                  dataKey="value"
                  fill={COLORS[index % COLORS.length]}
                  name={series.name || `Series ${index + 1}`}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )

      case "pie":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={transformedData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={120}
                fill="#8884d8"
                dataKey="value"
              >
                {transformedData.map((entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        )

      default:
        return (
          <div className="flex items-center justify-center h-40 text-muted-foreground">
            Unsupported chart type: {chart.type}
          </div>
        )
    }
  }

  return (
    <div className="space-y-3">
      <h3 className="font-medium">{chart.title || "Data Visualization"}</h3>
      {renderChart()}
    </div>
  )
}
