import { Card } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";

const data = [
  { time: "00:00", price: 42500, volume: 1200 },
  { time: "04:00", price: 43200, volume: 1450 },
  { time: "08:00", price: 42800, volume: 1350 },
  { time: "12:00", price: 44100, volume: 1820 },
  { time: "16:00", price: 43600, volume: 1680 },
  { time: "20:00", price: 44500, volume: 1950 },
  { time: "24:00", price: 45200, volume: 2100 },
];

export const PriceChart = () => {
  return (
    <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold">Bitcoin Price</h3>
          <p className="text-sm text-muted-foreground">Últimas 24 horas</p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold">$45,200</p>
          <p className="text-sm text-success flex items-center gap-1 justify-end">
            +$2,700 (6.35%)
          </p>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
          <XAxis 
            dataKey="time" 
            stroke="hsl(var(--muted-foreground))" 
            fontSize={12}
          />
          <YAxis 
            stroke="hsl(var(--muted-foreground))" 
            fontSize={12}
            domain={['dataMin - 500', 'dataMax + 500']}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'hsl(var(--popover))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
            }}
          />
          <Area 
            type="monotone" 
            dataKey="price" 
            stroke="hsl(var(--primary))" 
            strokeWidth={2}
            fill="url(#colorPrice)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  );
};
