function prepare_plot_export()
%PREPARE_PLOT_EXPORT Summary of this function goes here
%   Detailed explanation goes here

fig = gcf();
style = struct();
style.Version = '1';
style.Format = 'eps';
style.Preview = 'none';
style.Width = 'auto';
style.Height = 'auto';
style.Units = 'centimeters';
style.Color = 'rgb';
style.Background = 'w';          % '' = no change; 'w' = white background
style.FixedFontSize = '14';
style.ScaledFontSize = 'auto';
style.FontMode = 'fixed';
style.FontSizeMin = '8';
style.FixedLineWidth = '4';
style.ScaledLineWidth = 'auto';
style.LineMode = 'fixed';
style.LineWidthMin = '0.5';
style.FontName = 'auto';
style.FontWeight = 'auto';
style.FontAngle = 'auto';
style.FontEncoding = 'latin1';
style.PSLevel = '2';
style.Renderer = 'auto';
style.Resolution = 'auto';
style.LineStyleMap = 'none';
style.ApplyStyle = '0';
style.Bounds = 'loose';
style.LockAxes = 'on';
style.ShowUI = 'on';
style.SeparateText = 'off';
hgexport(fig,'temp_dummy',style,'applystyle', true);

end

