function mspca(filename)

level = 5;
wname = 'sym4';
npc = 'kais';

disp('Loading')
load(filename, 'A')

disp('First run')
[~, ~, npcA] = wmspca(A, level, wname, npc);

npcA(1:1) = zeros(1,1);
disp('Second run')
[L, ~, ~] = wmspca(A, level, wname, npcA);

disp('Saving')
save(filename, 'L')

end
