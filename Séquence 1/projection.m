clear all;
close all;

load eigenfaces;

nb_lignes = 400;
nb_colonnes = 300;
h = figure('Position',[0,0,0.67*L,0.67*H]);
figure('Name','RMSE en fonction du nombre de composantes principales','Position',[0.67*L,0,0.33*L,0.3*L]);

% Calcul de la RMSE entre images originales et images reconstruites :
RMSE_max = 0;

% Composantes principales des données d'apprentissage
C = X_centre*W;

 for q = 0:n-1
     composantes_principales = C(:, 1:q);		% q premières composantes principales
     premieres_eigenfaces = W(:, 1:q);		% q premières eigenfaces
     X_reconstruit = composantes_principales * premieres_eigenfaces' + individu_moyen;
     figure(1);
     set(h,'Name',['Utilisation des ' num2str(q) ' premieres composantes principales']);
     colormap gray;
     hold off;
     for k = 1:n
         subplot(nb_personnes_base, nb_postures_base,k);
         img = reshape(X_reconstruit(:,k), 400, 300);
         imagesc(img);
         hold on;
         axis image;
         axis off;
     end
     
     figure(2);
     hold on;

     RMSE = sqrt(mean(mean((X-X_reconstruit').^2)));
     RMSE_max = max(RMSE,RMSE_max);

     plot(q,RMSE,'r+','MarkerSize',8,'LineWidth',2);
     axis([0 n-1 0 1.1*RMSE_max]);
     set(gca,'FontSize',20);
     hx = xlabel('$q$','FontSize',30);
     set(hx,'Interpreter','Latex');
     ylabel('RMSE','FontSize',30);
     
     pause(0.01);
 end


 save projection;

 function Beta_chapeau = MCO(x, y)
    O = ones(length(x), 1);
    A = [x.^2 x.*y y.^2 x y O];
    A = [A ; 1 0 1 0 0 0];
    B = zeros(length(x), 1);
    B = [B; 1];
    Beta_chapeau = A\B;
end
